// Numerical model and kinematics, independent of the DOM and renderer.
export const mod1 = x => ((x % 1) + 1) % 1;
const add = (a, b) => a.map((x, i) => x + b[i]);
const scale = (a, s) => a.map(x => x * s);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
export function qmul(a, b) {
  const [x,y,z,w] = a, [u,v,s,t] = b;
  return [w*u+x*t+y*s-z*v, w*v-x*s+y*t+z*u, w*s+x*v-y*u+z*t, w*t-x*u-y*v-z*s];
}
export function qrotate(q, v) {
  const t = scale(cross(q.slice(0, 3), v), 2);
  return add(v, add(scale(t, q[3]), cross(q.slice(0, 3), t)));
}
export function rotvecQuaternion(r) {
  const theta = Math.hypot(...r), s = theta < 1e-8 ? 0.5 - theta*theta/48 : Math.sin(theta/2)/theta;
  return [...scale(r, s), Math.cos(theta/2)];
}
export function slerp(a, b, t) {
  let dot = a.reduce((s, x, i) => s + x*b[i], 0);
  if (dot < 0) { b = scale(b, -1); dot = -dot; }
  let q;
  if (dot > 0.9995) q = lerp(a, b, t);
  else {
    const angle = Math.acos(Math.min(1, dot)), sine = Math.sin(angle);
    q = add(scale(a, Math.sin((1-t)*angle)/sine), scale(b, Math.sin(t*angle)/sine));
  }
  return scale(q, 1/Math.hypot(...q));
}
export function rootPose(y, pathPos, yaw) {
  const q = rotvecQuaternion([0, 0, yaw]);
  return {position: add(pathPos, qrotate(q, y.slice(22, 25))),
    quaternion: qmul(q, rotvecQuaternion(y.slice(25, 28)))};
}
function network(x, layers) {
  for (const layer of layers) {
    x = layer.weight.map((row, r) => {
      const value = row.reduce((sum, w, i) => sum + w*x[i], layer.bias[r]);
      return layer.activation === 'tanh' ? Math.tanh(value) : value;
    });
  }
  return x;
}
export class Decoder {
  constructor(spec) { this.spec = spec; }
  mixture(context) {
    const s = this.spec;
    return network(context.map((x, i) => (x-s.context_mean[i])/s.context_scale[i]), s.layers);
  }
  basis(phase, derivative = false) {
    const h = [derivative ? 0 : 1];
    for (let k = 1; k <= 3; k++) {
      const angle = 2*Math.PI*k*phase;
      h.push(derivative ? -2*Math.PI*k*Math.sin(angle) : Math.cos(angle),
        derivative ? 2*Math.PI*k*Math.cos(angle) : Math.sin(angle));
    }
    return this.spec.waveforms.map(wave => wave.map(row => row.reduce((s,w,i) => s+w*h[i],0)));
  }
  decode(phase, z) {
    const b = this.basis(phase), s = this.spec;
    return b[0].map((v, d) => (v+z.reduce((sum,w,r) => sum+w*b[r+1][d],0))*s.target_scale[d]+s.target_mean[d]);
  }
  derivative(phase, z, phaseRate, zRate) {
    const b = this.basis(phase), db = this.basis(phase, true);
    return b[0].map((_, d) => ((db[0][d]+z.reduce((sum,w,r) => sum+w*db[r+1][d],0))*phaseRate
      + zRate.reduce((sum,w,r) => sum+w*b[r+1][d],0))*this.spec.target_scale[d]);
  }
  frequency(context) {
    const c = this.spec.cadence;
    if (!c) return 1/context[3];
    const output = network(context.slice(0,3).map((v,i) => (v-c.mean[i])/c.scale[i]), c.layers)[0];
    return c.min_hz+(c.max_hz-c.min_hz)/(1+Math.exp(-output));
  }
}
export function integratePath(position, yaw, command, dt) {
  const turn = command[2]*dt, half = turn/2, midpoint = yaw+half;
  const length = dt*(Math.abs(half) < 1e-8 ? 1-half*half/6 : Math.sin(half)/half);
  const [vx,vy] = command, c = Math.cos(midpoint), s = Math.sin(midpoint);
  return {position: [position[0]+length*(c*vx-s*vy), position[1]+length*(s*vx+c*vy), position[2]], yaw: yaw+turn};
}
export class Runtime {
  constructor(decoder, context, phase = 0) {
    this.decoder = decoder;
    this.time = 0; this.phase = mod1(phase); this.hz = decoder.frequency(context);
    this.z = decoder.mixture([...context.slice(0,3), 1/this.hz]);
    this.pathPos = [0,0,0]; this.pathYaw = 0;
  }
  clone() {
    const state = Object.create(Runtime.prototype);
    Object.assign(state, this, {z: [...this.z], pathPos: [...this.pathPos]});
    return state;
  }
  sample(zRate = [0,0,0,0]) {
    const y = this.decoder.decode(this.phase, this.z);
    return {time: this.time, phase: this.phase, contextPeriod: 1/this.hz, z: [...this.z], y,
      dy: this.decoder.derivative(this.phase, this.z, this.hz, zRate),
      root: rootPose(y, this.pathPos, this.pathYaw), pathPos: [...this.pathPos], pathYaw: this.pathYaw};
  }
  advance(context, dt) {
    if (!(dt > 0) || context.length !== 4 || !context.every(Number.isFinite) || context[3] <= 0)
      throw new Error('Invalid runtime command or timestep');
    const targetHz = this.decoder.frequency(context), delta = targetHz-this.hz;
    const ramp = Math.min(dt, Math.abs(delta)); // 1 cycle Hz/s, matching Python's default.
    const nextHz = this.hz+Math.sign(delta)*ramp;
    this.phase = mod1(this.phase+(this.hz+nextHz)*ramp/2+nextHz*(dt-ramp));
    this.hz = nextHz;
    const targetZ = this.decoder.mixture([...context.slice(0,3), 1/nextHz]);
    const tau = this.decoder.spec.tau_z_s, decay = Math.exp(-dt/tau);
    this.z = this.z.map((z,i) => targetZ[i]+(z-targetZ[i])*decay);
    const zRate = this.z.map((z,i) => (targetZ[i]-z)/tau);
    const path = integratePath(this.pathPos, this.pathYaw, context, dt);
    this.pathPos = path.position; this.pathYaw = path.yaw; this.time += dt;
    return this.sample(zRate);
  }
}
export function forwardKinematics(skeleton, joints, root) {
  const poses = [];
  for (const b of skeleton.bodies) {
    if (b.parent < 0) { poses.push({position: root.position, quaternion: root.quaternion}); continue; }
    const parent = poses[b.parent], [w,x,y,z] = b.quaternion_wxyz;
    let q = [x,y,z,w], pos = [...b.position];
    if (b.joint !== null) {
      const rotation = rotvecQuaternion(scale(b.axis, joints[b.joint]));
      pos = add(pos, qrotate(q, add(b.pivot, scale(qrotate(rotation, b.pivot), -1))));
      q = qmul(q, rotation);
    }
    poses.push({position: add(parent.position, qrotate(parent.quaternion, pos)),
      quaternion: qmul(parent.quaternion, q)});
  }
  return poses;
}
export class Clip {
  constructor(metadata, buffer, manifest) {
    this.meta = metadata;
    if (buffer.byteLength !== metadata.frames*manifest.stride*4) throw new Error('Incomplete clip file');
    this.data = new Float32Array(buffer); this.stride = manifest.stride;
    this.offsets = {}; let offset = 0;
    for (const [name,width] of manifest.fields) { this.offsets[name] = [offset,width]; offset += width; }
  }
  field(frame, key) {
    const [offset,width] = this.offsets[key], start = frame*this.stride+offset;
    return Array.from(this.data.subarray(start,start+width));
  }
  sample(time, decoder) {
    const index = Math.max(0,Math.min(this.meta.frames-1,time*this.meta.fps));
    const a = Math.floor(index), b = Math.min(a+1,this.meta.frames-1), t = index-a;
    const interp = key => lerp(this.field(a,key),this.field(b,key),t);
    const source = {position: interp('root_pos'), quaternion: slerp(this.field(a,'root_rot_xyzw'),this.field(b,'root_rot_xyzw'),t)};
    const valid = this.field(a,'valid')[0] > 0.5 && (t < 1e-7 || this.field(b,'valid')[0] > 0.5);
    const sample = {time, valid, source, joints: interp('dof_pos'), contacts: this.field(a,'contacts')};
    if (valid) {
      const pa = this.field(a,'phase')[0], pb = t < 1e-7 ? pa : this.field(b,'phase')[0];
      const phase = mod1(pa+mod1(pb-pa)*t), context = this.field(a,'context');
      const z = decoder.mixture(context), y = decoder.decode(phase,z);
      const pathPos = interp('path_pos'), pathYaw = interp('path_yaw')[0];
      Object.assign(sample, {phase,context,z,y,pathPos,pathYaw,learned:rootPose(y,pathPos,pathYaw)});
    }
    return sample;
  }
}
