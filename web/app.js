import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {DRACOLoader} from 'three/addons/loaders/DRACOLoader.js';
import {Decoder, Runtime, Clip, forwardKinematics, mod1} from './core.js?v=1a6dee2b1223';

const $ = id => document.getElementById(id);
const colours = {source:0x5488be, learned:0xd79b42};
const fixedStep = 1/60, rolloutDuration = 12;
const ui = {ready:false, mode:'compare', playing:true, time:0, clip:null, speed:1,
  history:[], accumulator:0, preset:false, traceDirty:true, loadingToken:0};
let decoder, manifest, skeleton, view, runtime, catalogue, modelId, predictionHistory=[];
const cache = new Map();
const models = new Map();
const timeText = t => `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const command = () => ['vx','vy','yaw','period'].map(id => Number($(id).value));
const duration = () => ui.mode==='compare' ? (ui.clip?.meta.duration_s ?? 1) : rolloutDuration;
async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  return response.json();
}

class View {
  constructor() {
    this.renderer = new THREE.WebGLRenderer({canvas:$('scene'),antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.shadowMap.enabled=true;
    this.renderer.shadowMap.type=THREE.PCFShadowMap;
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=1.2;
    this.scene=new THREE.Scene(); this.scene.background=new THREE.Color(0xf4f6f8);
    this.scene.fog=new THREE.Fog(0xf4f6f8,8,22);
    this.world=new THREE.Group(); this.world.rotation.x=-Math.PI/2;
    this.scene.add(this.world);
    this.camera=new THREE.PerspectiveCamera(36,1,0.02,100);
    this.controls=new OrbitControls(this.camera,$('scene'));
    this.controls.enableDamping=true; this.controls.dampingFactor=0.1;
    this.controls.minDistance=1.2; this.controls.maxDistance=18;
    this.controls.maxPolarAngle=Math.PI/2-0.015;
    this.scene.add(new THREE.HemisphereLight(0xffffff,0xb5bec8,2.2));
    this.sun=new THREE.DirectionalLight(0xffffff,3.1);
    this.sun.position.set(2,6,3); this.sun.castShadow=true;
    this.sun.shadow.mapSize.set(2048,2048); this.sun.shadow.normalBias=0.015;
    Object.assign(this.sun.shadow.camera,{left:-3,right:3,top:3,bottom:-3,near:0.1,far:15});
    this.scene.add(this.sun,this.sun.target);
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),
      new THREE.MeshStandardMaterial({color:0xf0f3f6,roughness:0.95}));
    floor.receiveShadow=true; this.world.add(floor);
    const grid=new THREE.GridHelper(200,400,0xd6dfe7,0xe2e8ee);
    grid.rotation.x=Math.PI/2; grid.position.z=0.0005; this.world.add(grid);
    this.robots={}; this.paths=new THREE.Group(); this.world.add(this.paths);
    this.markers=new THREE.Group(); this.world.add(this.markers);
    this.footMarkers=[0,1].map(()=>{
      const marker=new THREE.Mesh(new THREE.RingGeometry(0.045,0.057,40),
        new THREE.MeshBasicMaterial({color:0x5c9b82,side:THREE.DoubleSide,transparent:true,opacity:0.7}));
      marker.visible=false; this.markers.add(marker); return marker;
    });
    this.centre=new THREE.Vector3(0,0.45,0);
    this.resetCamera();
    this.controls.addEventListener('start',()=>{this.userOrbiting=true;});
    this.controls.addEventListener('end',()=>{this.userOrbiting=false;});
    new ResizeObserver(()=>this.resize()).observe($('stage'));
    this.resize();
  }
  resize() {
    const {width,height}=$('stage').getBoundingClientRect();
    this.renderer.setSize(width,height,false); this.camera.aspect=width/height;
    this.camera.updateProjectionMatrix(); ui.traceDirty=true;
  }
  resetCamera() {
    this.controls.target.copy(this.centre);
    this.camera.position.copy(this.centre).add(new THREE.Vector3(2.2,1.35,2.7));
    this.controls.update();
  }
  async loadRobots() {
    const loader=new GLTFLoader(), draco=new DRACOLoader();
    draco.setDecoderPath('./vendor/three/examples/jsm/libs/draco/gltf/');
    loader.setDRACOLoader(draco);
    const gltf=await loader.loadAsync('./assets/robot.glb');
    const map=new Map(skeleton.meshes.map(n=>[n.node,n.body]));
    const meshes=[];
    gltf.scene.traverse(obj=>{
      if (!obj.isMesh) return;
      let node=obj;
      while (node&&!map.has(node.name)) node=node.parent;
      if (!node) return;
      const geometry=obj.geometry.index?obj.geometry.toNonIndexed():obj.geometry.clone();
      geometry.deleteAttribute('normal'); geometry.computeVertexNormals();
      const material=Array.isArray(obj.material)?obj.material[0]:obj.material;
      meshes.push({geometry,body:map.get(node.name),colour:material.color.clone()});
    });
    if (!meshes.length) throw new Error('K1 visual mesh has no matching body names');
    for (const kind of ['source','learned']) {
      const group=new THREE.Group(); this.world.add(group);
      const parts=meshes.map(part=>{
        const colour=part.colour.clone().convertSRGBToLinear().lerp(new THREE.Color(colours[kind]),0.16);
        const mesh=new THREE.Mesh(part.geometry,new THREE.MeshStandardMaterial({color:colour,roughness:0.58,metalness:0.08}));
        mesh.castShadow=true; mesh.receiveShadow=true; group.add(mesh);
        return {mesh,body:part.body};
      });
      const badge=new THREE.Mesh(new THREE.RingGeometry(0.13,0.136,56),
        new THREE.MeshBasicMaterial({color:colours[kind],transparent:true,opacity:0.45,side:THREE.DoubleSide}));
      group.add(badge); this.robots[kind]={group,parts,badge};
    }
    draco.dispose();
  }
  pose(kind,joints,root,offset=0) {
    const robot=this.robots[kind]; robot.group.visible=true;
    const poses=forwardKinematics(skeleton,joints,root);
    for (const {mesh,body} of robot.parts) {
      const p=poses[body]; mesh.position.set(p.position[0],p.position[1]+offset,p.position[2]);
      mesh.quaternion.fromArray(p.quaternion);
    }
    robot.badge.position.set(root.position[0],root.position[1]+offset,0.003);
    return poses;
  }
  clearPaths() {
    for (const object of [...this.paths.children]) {
      object.geometry.dispose(); object.material.dispose(); this.paths.remove(object);
    }
  }
  addPath(points,colour,opacity=0.65) {
    if (points.length<2) return;
    const geometry=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(p[0],p[1],0.004)));
    const line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:colour,transparent:true,opacity}));
    this.paths.add(line);
  }
  setClipPaths(clip) {
    this.clearPaths(); const source=[],path=[]; let learned=[];
    for (let i=0;i<clip.meta.frames;i+=2) {
      const sample=clip.sample(i/clip.meta.fps,decoder);
      source.push([sample.source.position[0],sample.source.position[1]+0.55]);
      const pp=clip.field(i,'path_pos'); path.push(pp);
      if (sample.valid) learned.push([sample.learned.position[0],sample.learned.position[1]-0.55]);
      else if (learned.length) {this.addPath(learned,colours.learned); learned=[];}
    }
    this.addPath(source,colours.source); this.addPath(path,0x94a9a5,0.4);
    this.addPath(learned,colours.learned);
  }
  update(sample,elapsed) {
    let root;
    if (ui.mode==='compare') {
      const poses=this.pose('source',sample.joints,sample.source,0.55);
      this.robots.learned.group.visible=sample.valid;
      if (sample.valid) this.pose('learned',sample.y.slice(0,22),sample.learned,-0.55);
      root=sample.source;
      this.footMarkers.forEach((marker,i)=>{
        marker.visible=$('show-contacts').checked&&sample.contacts[i]>0.5;
        const foot=poses[skeleton.feet[i]].position;
        marker.position.set(foot[0],foot[1]+0.55,0.006);
      });
    } else {
      this.robots.source.group.visible=false;
      this.pose('learned',sample.y.slice(0,22),sample.root);
      root=sample.root; this.footMarkers.forEach(marker=>{marker.visible=false;});
    }
    this.paths.visible=$('show-path').checked;
    this.centre.set(root.position[0],0.43,-root.position[1]);
    if ($('follow').checked&&!this.userOrbiting) {
      const shift=this.centre.clone().sub(this.controls.target).multiplyScalar(1-Math.exp(-elapsed*5));
      this.controls.target.add(shift); this.camera.position.add(shift);
    }
    this.sun.position.copy(this.controls.target).add(new THREE.Vector3(2,6,3));
    this.sun.target.position.copy(this.controls.target);
    this.controls.update(); this.renderer.render(this.scene,this.camera);
  }
}

function setPlay(playing) {
  ui.playing=playing; $('play').setAttribute('aria-label',playing?'Pause playback':'Play playback');
  $('play').innerHTML=playing?'<svg viewBox="0 0 24 24" width="18" height="18"><path d="M8 5v14M16 5v14" stroke="currentColor" stroke-width="3"/></svg>':'<svg viewBox="0 0 24 24" width="18" height="18"><path d="m8 4 13 8-13 8z" fill="currentColor"/></svg>';
}
function updateSliderLabels() {
  for (const [id,unit] of [['vx','m/s'],['vy','m/s'],['yaw','rad/s'],['period','s']])
    $(`${id}-value`).value=`${Number($(id).value).toFixed(2)} ${unit}`;
}
function setCommand(values) {
  ['vx','vy','yaw','period'].forEach((id,i)=>{$(id).value=values[i];}); updateSliderLabels();
}
function updateModelReadout() {
  const entry=catalogue.models.find(m=>m.id===modelId);
  $('parameter-count').textContent=decoder.spec.parameters.toLocaleString();
  $('decoder-note').textContent=`${decoder.spec.parameters.toLocaleString()} parameters · ${entry.selection_note ?? 'selected'} seed ${entry.seed ?? '—'}`;
  $('state-note').textContent=decoder.spec.kind==='rff'
    ? 'Adjust commands while playing. Shared command inputs change smoothly.'
    : 'Adjust commands while playing. The shared mixture changes smoothly.';
  if (!ui.clip) return;
  const metrics=entry.metrics[ui.clip.meta.name]?.reconstruction ?? {};
  $('rmse').innerHTML=`${metrics.joint_rmse_rad===undefined?'—':metrics.joint_rmse_rad.toFixed(3)}<small>rad RMSE</small>`;
  $('foot-error').textContent=metrics.foot_position_rmse_m===undefined?'—':`${metrics.foot_position_rmse_m.toFixed(3)} m`;
  const learned=metrics.learned_contact_horizontal_speed_rms_m_s, source=metrics.source_contact_horizontal_speed_rms_m_s;
  $('contact-speed').textContent=learned===undefined?'—':`${learned.toFixed(2)} m/s · source ${source.toFixed(2)}`;
}
function selectDecoder(id) {
  if (id===modelId||!models.has(id)) return;
  modelId=id;decoder=models.get(id);ui.preset=false;
  for (const clip of cache.values())clip.traceSamples=null;
  if (ui.mode==='generate') resetRuntime();
  else {runtime=new Runtime(decoder,command());view.setClipPaths(ui.clip);}
  updateModelReadout();ui.traceDirty=true;
}
function recordCommand(values,time) {
  const last=ui.history.at(-1);
  if (!last||last.context.some((v,i)=>v!==values[i])) {
    ui.history=ui.history.filter(c=>c.time<time-1e-8);
    ui.history.push({time,context:[...values]});
  }
}
function resetRuntime(phase=0,clear=true) {
  ui.time=0;ui.accumulator=0;
  if (clear) ui.history=[{time:0,context:command()}];
  runtime=new Runtime(decoder,ui.history[0].context,phase);
  predictionHistory=[runtime.sample()]; view.clearPaths(); ui.traceDirty=true;
}
function commandAt(time) {
  let result=ui.history[0].context;
  for (const c of ui.history) {if (c.time<=time+1e-10) result=c.context;else break;}
  return result;
}
function advanceRuntime(targetTime) {
  while (runtime.time+fixedStep<=targetTime+1e-8) {
    const sample=runtime.advance(commandAt(runtime.time),fixedStep);
    predictionHistory.push(sample);
  }
  ui.time=runtime.time;
}
function scrubRuntime(time) {
  runtime=new Runtime(decoder,ui.history[0].context);predictionHistory=[runtime.sample()];
  advanceRuntime(time);setCommand(commandAt(time));ui.accumulator=0;ui.traceDirty=true;
}
function setMode(mode) {
  if (!ui.ready||mode===ui.mode) return;
  ui.mode=mode;ui.preset=false;
  for (const id of ['compare','generate']) {
    const selected=mode===id;
    $(`${id}-tab`).setAttribute('aria-selected',String(selected));
    $(`${id}-tab`).tabIndex=selected?0:-1;$(`${id}-panel`).hidden=!selected;
  }
  $('source-legend').hidden=mode==='generate'; $('show-contacts').disabled=mode==='generate';
  $('timeline').max=duration();
  $('trace-note').textContent=mode==='compare'?'Recorded and decoded joint angles across the clip.':'Generated joint angles and the evolving command history.';
  if (mode==='generate') {
    $('stage-mode').textContent='Free rollout · uniform phase clock';
    $('stage-detail').textContent='Integrated root path · smooth shared inputs';
    resetRuntime();
    view.centre.set(0,0.43,0);
  } else {
    $('stage-detail').textContent='Shared world path · fixed display separation';
    ui.time=0;ui.accumulator=0;view.setClipPaths(ui.clip);
  }
  setPlay(true);ui.traceDirty=true;view.resetCamera();
}
async function selectClip(name) {
  const token=++ui.loadingToken, meta=manifest.clips.find(c=>c.name===name);
  if (!meta) throw new Error('Unknown clip');
  $('clip').disabled=true;$('play').disabled=true;
  if (!cache.has(name)) {
    const response=await fetch(`./assets/${meta.file}`);
    if (!response.ok) throw new Error(`Clip download failed (${response.status})`);
    cache.set(name,new Clip(meta,await response.arrayBuffer(),manifest));
  }
  if (token!==ui.loadingToken) return;
  ui.clip=cache.get(name);ui.time=0;ui.accumulator=0;$('timeline').max=duration();
  $('split-badge').textContent={train:'Training',dev:'Development',test:'Held out'}[meta.split];
  $('clip-info').textContent=`${meta.frames.toLocaleString()} frames · ${meta.fps} Hz · ${meta.duration_s.toFixed(1)} seconds`;
  updateModelReadout();
  $('coverage').textContent=`${meta.valid_frames.toLocaleString()} / ${meta.frames.toLocaleString()} frames have cycle labels.`;
  $('phase-note').textContent=meta.phase_source.includes('fallback')?'Phase uses a foot-height fallback in this clip.':'Left-foot strikes anchor each complete gait cycle.';
  view.setClipPaths(ui.clip);ui.traceDirty=true;$('clip').disabled=false;$('play').disabled=false;
  const root=ui.clip.sample(0,decoder).source;
  view.centre.set(root.position[0],0.43,-root.position[1]);view.resetCamera();
}
function updateReadout(sample) {
  $('timeline').value=ui.time;$('time-label').textContent=`${timeText(ui.time)} / ${timeText(duration())}`;
  const valid=ui.mode==='generate'||sample.valid;
  const phase=valid?sample.phase:null;
  $('phase-value').textContent=phase===null?'—':phase.toFixed(2);
  $('phase-ring').style.strokeDasharray=`${phase??0} 1`;
  const hz=ui.mode==='generate'?runtime.hz:(valid?1/sample.context[3]:null);
  $('cycle-hz').textContent=hz===null?'Unlabelled frame':`${hz.toFixed(2)} cycles/s`;
  if (ui.mode==='compare') {
    $('stage-mode').textContent=valid?'Observed-phase reconstruction':'Source frame · no cycle label';
    ['left','right'].forEach((side,i)=>$(`contact-${side}`).classList.toggle('active',sample.contacts[i]>0.5));
  }
}
function drawTrace() {
  const canvas=$('trace'), rect=canvas.getBoundingClientRect(), dpr=Math.min(devicePixelRatio,2);
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const width=rect.width,height=rect.height,joint=Number($('joint').value),end=duration();
  const source=[],learned=[];
  if (ui.mode==='compare'&&ui.clip) {
    if (!ui.clip.traceSamples) {
      const step=Math.max(1,Math.floor(ui.clip.meta.frames/1000));
      ui.clip.traceSamples=[];
      for(let i=0;i<ui.clip.meta.frames;i+=step)
        ui.clip.traceSamples.push(ui.clip.sample(i/ui.clip.meta.fps,decoder));
    }
    for (const s of ui.clip.traceSamples) {
      source.push([s.time,s.joints[joint]]);learned.push([s.time,s.valid?s.y[joint]:NaN]);
    }
  } else predictionHistory.forEach(s=>learned.push([s.time,s.y[joint]]));
  const values=[...source,...learned].map(p=>p[1]).filter(Number.isFinite);
  const min=Math.min(...values,0)-0.08,max=Math.max(...values,0.1)+0.08;
  const left=43,right=12,top=12,bottom=26,w=width-left-right,h=height-top-bottom;
  const x=t=>left+t/end*w,y=v=>top+(max-v)/(max-min)*h;
  ctx.font='9px system-ui';ctx.fillStyle='#8b99a5';ctx.lineWidth=1;
  for (let i=0;i<3;i++) {
    const v=min+(max-min)*i/2,py=y(v);
    ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(width-right,py);ctx.strokeStyle='#edf1f5';ctx.stroke();
    ctx.textAlign='right';ctx.fillText(v.toFixed(1),left-9,py+3);
  }
  ctx.fillText('rad',left-9,top-2);
  for (let i=0;i<5;i++) {ctx.textAlign='center';ctx.fillText(`${(end*i/4).toFixed(1)} s`,x(end*i/4),height-7);}
  function line(points,colour) {
    ctx.beginPath();let started=false;
    for (const [t,v] of points) {
      if (!Number.isFinite(v)) {started=false;continue;}
      if (!started) {ctx.moveTo(x(t),y(v));started=true;}else ctx.lineTo(x(t),y(v));
    }
    ctx.strokeStyle=colour;ctx.lineWidth=1.4;ctx.stroke();
  }
  line(source,'#5488be');line(learned,'#c79243');
  ctx.beginPath();ctx.moveTo(x(ui.time),top);ctx.lineTo(x(ui.time),height-bottom);
  ctx.strokeStyle='#344e6880';ctx.lineWidth=1;ctx.stroke();
  ui.traceDirty=false;
}
function download(payload,name) {
  if (ui.exportUrl) URL.revokeObjectURL(ui.exportUrl);
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload)],{type:'application/json'}));
  ui.exportUrl=url;
  $('export-download').href=url;$('export-download').download=name;
  $('export-filename').textContent=name;
  $('export-details').textContent=`${payload.frames.length.toLocaleString()} frames · ${payload.fps} Hz · world coordinates`;
  $('export-json').value=JSON.stringify(payload);
  $('copy-status').textContent='';$('export-dialog').showModal();
}
function exportMotion() {
  const common={schema_version:1,joint_names:decoder.spec.joint_names,units:{position:'m',angle:'rad',time:'s'},
    model_source_sha256:decoder.spec.source_sha256,decoder_id:modelId,
    decoder_kind:decoder.spec.kind,decoder_parameters:decoder.spec.parameters,
    runtime_state:decoder.spec.runtime_state,foot_constraints:false};
  if (ui.mode==='generate') {
    const preview=new Runtime(decoder,ui.history[0].context),frames=[preview.sample()];
    for (let i=0;i<rolloutDuration/fixedStep;i++) frames.push(preview.advance(commandAt(preview.time),fixedStep));
    download({...common,mode:'free_rollout',fps:60,commands:ui.history,tau_z_s:decoder.spec.tau_z_s,
      frames:frames.map(s=>({time:s.time,phase:s.phase,coordinates:s.y,coordinate_velocity:s.dy,
        root_pos:s.root.position,root_rot_xyzw:s.root.quaternion,path_pos:s.pathPos,path_yaw:s.pathYaw}))},'oscillator-rollout.json');
  } else {
    const frames=[];
    for (let i=0;i<ui.clip.meta.frames;i++) {
      const s=ui.clip.sample(i/ui.clip.meta.fps,decoder);
      frames.push({time:s.time,valid:s.valid,phase:s.valid?s.phase:null,context:s.valid?s.context:null,
        source_dof_pos:s.joints,source_root_pos:s.source.position,source_root_rot_xyzw:s.source.quaternion,
        coordinates:s.valid?s.y:null,root_pos:s.valid?s.learned.position:null,
        root_rot_xyzw:s.valid?s.learned.quaternion:null,source_contacts:s.contacts,
        path_pos:ui.clip.field(i,'path_pos'),path_yaw:ui.clip.field(i,'path_yaw')[0]});
    }
    download({...common,mode:'observed_phase_reconstruction',clip:ui.clip.meta.name,
      fps:ui.clip.meta.fps,root_path:'saved_offline_fit',frames},`${ui.clip.meta.name}-reconstruction.json`);
  }
}
function bindControls() {
  $('decoder').addEventListener('change',()=>selectDecoder($('decoder').value));
  $('compare-tab').addEventListener('click',()=>setMode('compare'));
  $('generate-tab').addEventListener('click',()=>setMode('generate'));
  document.querySelector('.mode-tabs').addEventListener('keydown',e=>{
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
    e.preventDefault();setMode(e.key==='Home'?'compare':e.key==='End'?'generate':ui.mode==='compare'?'generate':'compare');
    $(`${ui.mode}-tab`).focus();
  });
  $('play').addEventListener('click',()=>setPlay(!ui.playing));
  $('restart').addEventListener('click',()=>{ui.preset=false;if(ui.mode==='generate')resetRuntime();else ui.time=0;ui.traceDirty=true;});
  $('clip').addEventListener('change',()=>selectClip($('clip').value).catch(showError));
  $('speed').addEventListener('change',()=>{ui.speed=Number($('speed').value);});
  $('timeline').addEventListener('input',()=>{
    setPlay(false);const time=Number($('timeline').value);
    if(ui.mode==='generate')scrubRuntime(time);else ui.time=time;ui.traceDirty=true;
  });
  for (const id of ['vx','vy','yaw','period']) $(id).addEventListener('input',()=>{
    updateSliderLabels();ui.preset=false;recordCommand(command(),runtime.time);ui.traceDirty=true;
    $('command-note').textContent='Commands condition the pose generator; physical tracking is not simulated.';
  });
  $('walking').addEventListener('click',()=>{ui.preset=false;setCommand([0.8,0,0,0.9]);recordCommand(command(),runtime.time);});
  $('turning').addEventListener('click',()=>{ui.preset=false;setCommand([0.8,0,0.5,0.9]);recordCommand(command(),runtime.time);});
  $('transition').addEventListener('click',()=>{
    setCommand([0.8,0,0,0.9]);resetRuntime();
    ui.history=[{time:0,context:[0.8,0,0,0.9]},{time:3,context:[1.3,0,0.35,0.7]},
      {time:7,context:[0.8,0.2,0,0.9]}];ui.preset=true;setPlay(true);
    $('command-note').textContent='12-second transition: faster turn at 3 s, lateral motion at 7 s.';
  });
  $('joint').addEventListener('change',()=>{ui.traceDirty=true;});
  $('reset-camera').addEventListener('click',()=>view.resetCamera());
  $('reframe').addEventListener('click',()=>{
    const delta=view.centre.clone().sub(view.controls.target);
    view.controls.target.add(delta);view.camera.position.add(delta);view.controls.update();
  });
  $('export').addEventListener('click',exportMotion);
  $('export-close').addEventListener('click',()=>$('export-dialog').close());
  $('export-copy').addEventListener('click',async()=>{
    try {await navigator.clipboard.writeText($('export-json').value);$('copy-status').textContent='Copied to clipboard.';}
    catch {$('export-json').focus();$('export-json').select();$('copy-status').textContent='Select the text and copy it with your keyboard.';}
  });
  $('scene').addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();setPlay(!ui.playing);}});
  $('scene').tabIndex=0;
  document.addEventListener('visibilitychange',()=>{ui.accumulator=0;});
}
function showError(error) {
  console.error(error);$('loading').hidden=false;$('loading').replaceChildren();
  const div=document.createElement('div');div.className='error-message';
  const title=document.createElement('strong');title.textContent='The viewer could not load.';
  const detail=document.createElement('p');detail.textContent=error.message;
  const help=document.createElement('p');help.textContent='Serve the web folder over HTTP, then reload. See the README for the local preview command.';
  div.append(title,detail,help);$('loading').append(div);
}
async function start() {
  const [catalog,data,rig]=await Promise.all([json('./assets/models.json'),json('./assets/manifest.json'),json('./assets/skeleton.json')]);
  catalogue=catalog;manifest=data;skeleton=rig;
  await Promise.all(catalogue.models.map(async entry=>{
    const spec=await json(`./assets/${entry.file}`);
    if(JSON.stringify(spec.joint_names)!==JSON.stringify(skeleton.joint_names)) throw new Error('Model and K1 joint orders disagree');
    models.set(entry.id,new Decoder(spec));
  }));
  modelId=catalogue.models[0].id;decoder=models.get(modelId);view=new View();
  $('decoder').replaceChildren(...catalogue.models.map(entry=>{
    const option=document.createElement('option');option.value=entry.id;option.textContent=entry.label;return option;
  }));$('decoder').disabled=false;
  $('clip').replaceChildren(...manifest.clips.map(meta=>{
    const option=document.createElement('option');option.value=meta.name;
    option.textContent=`${meta.name} · ${meta.split==='test'?'held out':meta.split}`;return option;
  }));
  $('clip').value='walk3_subject2_04';
  $('joint').replaceChildren(...decoder.spec.joint_names.map((name,index)=>{
    const option=document.createElement('option');option.value=index;option.textContent=name.replaceAll('_',' ');return option;
  }));$('joint').value=10;
  await Promise.all([view.loadRobots(),selectClip($('clip').value)]);
  runtime=new Runtime(decoder,command());ui.history=[{time:0,context:command()}];
  $('loading').hidden=true;$('export').disabled=false;ui.ready=true;bindControls();
  let previous=performance.now(),lastTrace=0;
  function frame(now) {
    const elapsed=Math.min((now-previous)/1000,0.05);previous=now;
    if(!document.hidden) {
      if(ui.playing&&!$('clip').disabled) {
        if(ui.mode==='compare')ui.time=(ui.time+elapsed*ui.speed)%duration();
        else {
          ui.accumulator+=elapsed*ui.speed;
          while(ui.accumulator>=fixedStep) {
            if(runtime.time>=rolloutDuration-fixedStep/2) {
              const remaining=ui.accumulator;
              resetRuntime(0,false);ui.accumulator=remaining;
            }
            predictionHistory.push(runtime.advance(commandAt(runtime.time),fixedStep));
            ui.accumulator-=fixedStep;ui.time=runtime.time;
          }
          if(ui.preset)setCommand(commandAt(runtime.time));
          if(now-lastTrace>100){ui.traceDirty=true;lastTrace=now;}
          if(predictionHistory.length%15===0) {
            view.clearPaths();view.addPath(predictionHistory.map(s=>s.pathPos),0x94a9a5,0.4);
            view.addPath(predictionHistory.map(s=>s.root.position),colours.learned);
          }
        }
      }
      const sample=ui.mode==='compare'?ui.clip.sample(ui.time,decoder):predictionHistory.at(-1);
      view.update(sample,elapsed);updateReadout(sample);
      if(ui.mode==='compare'&&ui.playing&&now-lastTrace>80){ui.traceDirty=true;lastTrace=now;}
      if(ui.traceDirty)drawTrace();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  // Read-only diagnostics for local validation, without exposing mutable runtime state.
  window.oscillatorViewer={get status(){return {ready:ui.ready,mode:ui.mode,time:ui.time,
    clip:ui.clip.meta.name,phase:runtime.phase,frames:predictionHistory.length};}};
}
start().catch(showError);
