import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Decoder, Runtime, Clip, forwardKinematics, integratePath, slerp} from '../web/core.js';
const json = name => JSON.parse(fs.readFileSync(new URL(`../web/assets/${name}.json`,import.meta.url)));
const model = new Decoder(json('model')), validation = json('validation');
const catalogue = json('models');
function close(a,b,tolerance=5e-6) {
  assert.equal(a.length,b.length);
  a.forEach((v,i)=>assert.ok(Math.abs(v-b[i]) < tolerance, `${i}: ${v} != ${b[i]}`));
}
test('browser decoder and derivatives match the saved PyTorch model',()=>{
  for (const c of validation.cases) {
    close(model.mixture(c.context),c.z);
    close(model.decode(c.phase,model.mixture(c.context)),c.y);
    close(model.derivative(c.phase,c.z,c.phase_rate,c.z_rate),c.dy,1e-5);
    close(model.decode(c.phase+1,c.z),model.decode(c.phase,c.z),1e-12);
  }
});
for (const entry of catalogue.models) {
  const decoder = new Decoder(JSON.parse(fs.readFileSync(new URL(`../web/assets/${entry.file}`,import.meta.url))));
  const fixtures = JSON.parse(fs.readFileSync(new URL(`../web/assets/${entry.validation}`,import.meta.url)));
  test(`${entry.label}: inference, derivatives, periodicity and transitions match Python`,()=>{
    for (const c of fixtures.cases) {
      close(decoder.mixture(c.context),c.z,1e-10);
      close(decoder.decode(c.phase,c.z),c.y,1e-10);
      close(decoder.derivative(c.phase,c.z,c.phase_rate,c.z_rate),c.dy,1e-6);
      close(decoder.decode(c.phase+1,c.z),c.y,1e-10);
      close(decoder.derivative(c.phase+1,c.z,c.phase_rate,c.z_rate),
        decoder.derivative(c.phase,c.z,c.phase_rate,c.z_rate),1e-9);
    }
    const runtime = new Runtime(decoder,[0.8,0,0,0.9]);
    for(let i=0;i<60;i++) {
      const sample=runtime.advance(i<30?[0.8,0,0,0.9]:[1.2,0.2,0.3,0.7],1/60);
      const expected=fixtures.runtime.find(frame=>frame.step===i+1);
      if(expected) {close(sample.y,expected.y,1e-10);close(runtime.z,expected.z,1e-9);close(sample.root.position,expected.root_pos,1e-10);}
    }
    const copy=runtime.clone(), z=[...runtime.z], path=[...runtime.pathPos];
    copy.advance([1,0.2,0.5,0.8],0.2);
    assert.deepEqual(runtime.z,z);assert.deepEqual(runtime.pathPos,path);
  });
}
test('runtime transitions and world roots match Python',()=>{
  const r = new Runtime(model,[0.8,0,0,0.9]);
  for (let i=0;i<60;i++) {
    const sample = r.advance(i<30?[0.8,0,0,0.9]:[1.2,0.2,0.3,0.7],1/60);
    const expected = validation.runtime.find(c=>c.step===i+1);
    if (expected) {
      close(r.z,expected.z); close(sample.y,expected.y);
      close(sample.root.position,expected.root_pos);
      close(sample.root.quaternion,expected.root_rot_xyzw);
      assert.ok(Math.abs(r.phase-expected.phase)<1e-6);
    }
  }
  const clone = r.clone(), z = [...r.z], path = [...r.pathPos];
  clone.advance([1,0,1,1],0.1);
  assert.deepEqual(r.z,z); assert.deepEqual(r.pathPos,path);
});
test('clip decoding masks excluded frames and retains the world root',()=>{
  const manifest = json('manifest');
  for (const meta of manifest.clips) {
    const bytes = fs.readFileSync(new URL(`../web/assets/${meta.file}`,import.meta.url));
    const clip = new Clip(meta,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),manifest);
    const frame = clip.sample(0,model);
    assert.equal(frame.valid,clip.field(0,'valid')[0]>0.5);
    close(frame.source.position,clip.field(0,'root_pos'));
    let samples = 0;
    for (let i=0;i<meta.frames;i++) {
      const s = clip.sample(i/meta.fps,model);
      if (s.valid) {
        samples++;
        assert.ok(s.y.every(Number.isFinite));
        assert.ok(s.phase >= 0 && s.phase < 1);
      }
    }
    assert.equal(samples,meta.valid_frames);
  }
});
test('K1 forward kinematics maps all joints and foot bodies',()=>{
  const skeleton = json('skeleton');
  assert.deepEqual(skeleton.joint_names,model.spec.joint_names);
  const pose = {position:[1,2,0.55],quaternion:[0,0,0,1]};
  const frames = forwardKinematics(skeleton,new Array(22).fill(0),pose);
  close(frames[0].position,pose.position);
  close(frames[skeleton.feet[0]].position,[0.99819706,2.0962,0.03631],1e-7);
  close(frames[skeleton.feet[1]].position,[0.99819706,1.9038,0.03631],1e-7);
  close(integratePath([0,0,0],0,[1,0,1],Math.PI/2).position,[1,1,0],1e-12);
  close(slerp([0,0,0,1],[0,0,0,-1],0.5),[0,0,0,1],1e-12);
  for (const c of validation.kinematics) {
    const fk = forwardKinematics(skeleton,c.joints,c);
    skeleton.feet.forEach((body,i)=>close(fk[body].position,c.feet[i],2e-6));
  }
});
