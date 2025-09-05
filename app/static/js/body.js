
class Body {
    constructor(id, name, radiusKm, massKg) {
        this.id = id;
        this.name = name;
        this.radiusKm = radiusKm;
        this.massKg = massKg;

        this.lastMeters = new THREE.Vector3();

        const tex = makeCircleTexture("#bbb", 128, 3);
        // keep original texture so flashColor can restore it
        this.baseMap = tex;
        this.material = new THREE.SpriteMaterial({ map: tex, transparent: true });
        this.sprite = new THREE.Sprite(this.material);
        this.sprite.userData.ref = this; // for picking
        scene.add(this.sprite);

        this.baseScale = 0.2;

        // interpolation state (world coordinates)
        this._prevWorld = new THREE.Vector3();   // start of interpolation
        this._nextWorld = new THREE.Vector3();   // target of interpolation
        this._lerpStart = 0;
        this._lerpDur = 0;

        // trail (orbit trace)
        this.trail = []; // array of THREE.Vector3
        // Buffer geometry for line
        const posArr = new Float32Array(TRAIL_MAX * 3);
        this.trailGeometry = new THREE.BufferGeometry();
        this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
        this.trailGeometry.setDrawRange(0, 0);
        this.trailMaterial = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.18 });
        this.trailLine = new THREE.Line(this.trailGeometry, this.trailMaterial);
        scene.add(this.trailLine);

        // flash color state
        this._flashTimeout = null;
        this._currentTempMap = null;
    }

    setScale(worldSize) {
        this.baseScale = worldSize;
        this.sprite.scale.set(worldSize, worldSize, 1);
    }

  setTrailFromHistory(historyArr) {
    // console.log("Setting history of length", historyArr.length);
    if (!historyArr || historyArr.length === 0) return;
    this.trail.length = 0;
    for (let i = 0; i < historyArr.length && this.trail.length < TRAIL_MAX; ++i) {
      const e = historyArr[i];
      let hx, hy, hz;
      if (Array.isArray(e) && e.length >= 3) {
        hx = e[0]; hy = e[1]; hz = e[2];
      } else if (e && typeof e === 'object' && 'x' in e && 'y' in e && 'z' in e) {
        hx = e.x; hy = e.y; hz = e.z;
      } else {
        continue;
      }
      const wx = (hx - sceneCenter.x) * sceneScale;
      const wy = (hy - sceneCenter.y) * sceneScale;
      const wz = (hz - sceneCenter.z) * sceneScale;
      this.trail.push(new THREE.Vector3(wx, wy, wz));
    }
    // console.log(`Set trail with ${this.trail.length} points`);
    if (this.trail.length === 0) return;

    const last = this.trail[this.trail.length - 1];
    this.sprite.position.copy(last);
    this._prevWorld.copy(last);
    this._nextWorld.copy(last);
    this._lerpStart = 0;
    this._lerpDur = 0;

    this._updateTrailGeometry();
  }

  // Immediately set position (used on first frame / creation)
  setImmediatePositionMeters(mx, my, mz) {
    this.lastMeters.set(mx, my, mz);
    const wx = (mx - sceneCenter.x) * sceneScale;
    const wy = (my - sceneCenter.y) * sceneScale;
    const wz = (mz - sceneCenter.z) * sceneScale;
    this._prevWorld.set(wx, wy, wz);
    this._nextWorld.copy(this._prevWorld);
    this._lerpStart = 0;
    this._lerpDur = 0;
    this.sprite.position.copy(this._nextWorld);

    // initialize trail with the current position
    this.trail.length = 0;
    for (let i = 0; i < Math.min(8, TRAIL_MAX); ++i) this.trail.push(this._nextWorld.clone());
    this._updateTrailGeometry();
  }

  // Schedule a smooth move to a new world position over durationMs
  moveToMeters(mx, my, mz, durationMs = API_POLL_MS) {
    this.lastMeters.set(mx, my, mz);
    const wx = (mx - sceneCenter.x) * sceneScale;
    const wy = (my - sceneCenter.y) * sceneScale;
    const wz = (mz - sceneCenter.z) * sceneScale;
    // shift current displayed position to prev, target becomes next
    this._prevWorld.copy(this.sprite.position);
    this._nextWorld.set(wx, wy, wz);
    this._lerpStart = performance.now();
    this._lerpDur = Math.max(50, durationMs); // avoid zero duration
    // push a trail sample at the start of each move (keeps orbit trace)
    this._pushTrailSample(this._prevWorld.clone());
  }

  // called each frame to advance interpolation
  updateLerp(now) {
    if (this._lerpDur > 0 && this._lerpStart > 0) {
      const t = Math.min(1, (now - this._lerpStart) / this._lerpDur);
      this.sprite.position.lerpVectors(this._prevWorld, this._nextWorld, t);
      // while animating, optionally push trailing intermediate points occasionally
      // don't push every frame to avoid flooding the trail
      if (Math.random() < 0.02) this._pushTrailSample(this.sprite.position.clone());
      if (t >= 1) {
        this._lerpStart = 0;
        // ensure final push
        this._pushTrailSample(this._nextWorld.clone());
      }
    }
    // make sure trail geometry follows sprite updates when hovered state changes color/opacity handled externally
  }

  _pushTrailSample(worldVec) {
    this.trail.push(worldVec);
    if (this.trail.length > TRAIL_MAX) this.trail.shift();
    this._updateTrailGeometry();
  }

  _updateTrailGeometry() {
    const drawCount = this.trail.length;
    const attr = this.trailGeometry.getAttribute('position');
    for (let i = 0; i < drawCount; ++i) {
      const v = this.trail[i];
      attr.setXYZ(i, v.x, v.y, v.z);
    }
    // zero out remaining slots to avoid artifacts
    for (let i = drawCount; i < TRAIL_MAX; ++i) attr.setXYZ(i, 0, 0, 0);
    attr.needsUpdate = true;
    this.trailGeometry.setDrawRange(0, drawCount);
  }

  setPositionMeters(mx, my, mz) {
    // legacy: immediate set (kept for compatibility)
    this.setImmediatePositionMeters(mx, my, mz);
  }

  setHovered(on) {
    if (on) {
      this.sprite.scale.set(this.baseScale * HOVER_SCALE, this.baseScale * HOVER_SCALE, 1);
      this.trailMaterial.color.set(0x000000);
      this.trailMaterial.opacity = 1.0;
    } else {
      this.sprite.scale.set(this.baseScale, this.baseScale, 1);
      this.trailMaterial.color.set(0x000000);
      this.trailMaterial.opacity = 0.20;
    }
    this.trailMaterial.needsUpdate = true;
  }

  // Temporarily tint / recolor the body's sprite by swapping its texture.
  // color: CSS color string (e.g. "#000" or "rgba(0,0,0,1)")
  // durationMs: how long before restoring (default: FLASH_DURATION_MS)
  flashColor(color, durationMs = FLASH_DURATION_MS) {
    // clear any pending restore
    if (this._flashTimeout) {
      clearTimeout(this._flashTimeout);
      this._flashTimeout = null;
    }
    // dispose any previous temp map
    if (this._currentTempMap) {
      try { this._currentTempMap.dispose(); } catch (e) {}
      this._currentTempMap = null;
    }

    // create a temporary circle texture in the requested color
    const tmp = makeCircleTexture(color, 128, 3);
    this._currentTempMap = tmp;
    this.material.map = tmp;
    this.material.needsUpdate = true;

    // schedule restore
    this._flashTimeout = setTimeout(() => {
      this._flashTimeout = null;
      // restore original map (guard if removed)
      try {
        this.material.map = this.baseMap;
        this.material.needsUpdate = true;
      } catch (e) {}
      // dispose temp map
      try { tmp.dispose(); } catch (e) {}
      this._currentTempMap = null;
    }, durationMs);
  }
}