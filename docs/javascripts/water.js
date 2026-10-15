// Liquid water over the hero sky: a height-field wave simulation on the GPU, rendered as a refracting surface.
// The sky is redrawn in the shader from the --gn-sky-* stops in stylesheets/extra.css, so the swap is invisible;
// the CSS grain stays on top. Without WebGL2 float render targets, or with reduced motion, the static CSS hero
// is left untouched.

(() => {
  const QUAD = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

  const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

  // one step of the wave equation: r = height, g = velocity
  const STEP = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform vec2 u_texel;
uniform float u_damping;
in vec2 v_uv;
out vec4 o;
void main() {
  vec4 s = texture(u_state, v_uv);
  float avg = (
    texture(u_state, v_uv + vec2(u_texel.x, 0.0)).r +
    texture(u_state, v_uv - vec2(u_texel.x, 0.0)).r +
    texture(u_state, v_uv + vec2(0.0, u_texel.y)).r +
    texture(u_state, v_uv - vec2(0.0, u_texel.y)).r
  ) * 0.25;
  s.g += (avg - s.r) * 2.0;
  s.g *= u_damping;
  s.r += s.g;
  s.r *= 0.9995;
  o = s;
}`;

  // a smooth cosine-shaped drop
  const DROP = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform vec2 u_center;
uniform float u_radius;
uniform float u_strength;
uniform float u_aspect;
in vec2 v_uv;
out vec4 o;
void main() {
  vec4 s = texture(u_state, v_uv);
  vec2 d = (v_uv - u_center) * vec2(u_aspect, 1.0);
  float k = max(0.0, 1.0 - length(d) / u_radius);
  s.r += (0.5 - cos(k * 3.14159265) * 0.5) * u_strength;
  o = s;
}`;

  // the sky of .gn-hero, refracted by the surface normal, with a soft specular sheen
  const RENDER = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform vec2 u_texel;
uniform vec2 u_size;
in vec2 v_uv;
out vec4 o;

// the seven --gn-sky-* stops, and the sun's three
uniform vec3 u_sky[7];
uniform vec3 u_sun[3];

// stop positions of --gn-sky in extra.css
const float POS[7] = float[7](0.0, 0.20, 0.42, 0.58, 0.76, 0.90, 1.0);

vec3 sky(vec2 p) {
  // linear-gradient(172deg, ...) in CSS pixels, y pointing down
  float a = radians(172.0);
  vec2 dir = vec2(sin(a), -cos(a));
  float len = abs(u_size.x * dir.x) + abs(u_size.y * dir.y);
  float t = clamp(dot(p - 0.5 * u_size, dir) / len + 0.5, 0.0, 1.0);
  vec3 col = u_sky[0];
  for (int i = 1; i < 7; i++) col = mix(col, u_sky[i], clamp((t - POS[i - 1]) / (POS[i] - POS[i - 1]), 0.0, 1.0));

  // the low sun: .gn-hero::before, a radial gradient in a circle below the bottom edge
  float D = 0.62 * u_size.x;
  vec2 center = vec2(0.5 * u_size.x, 1.62 * u_size.y - 0.5 * D);
  float d = length(p - center) / (0.5 * D * 1.41421356);
  vec4 s0 = vec4(u_sun[0] * 0.8, 0.8);
  vec4 s1 = vec4(u_sun[1] * 0.4, 0.4);
  vec4 s2 = vec4(u_sun[2] * 0.149, 0.149);
  vec4 sun = mix(s0, s1, clamp(d / 0.24, 0.0, 1.0));
  sun = mix(sun, s2, clamp((d - 0.24) / 0.22, 0.0, 1.0));
  sun = mix(sun, vec4(0.0), clamp((d - 0.46) / 0.22, 0.0, 1.0));
  return sun.rgb + col * (1.0 - sun.a);
}

float height(vec2 uv) { return texture(u_state, uv).r; }

void main() {
  float hx = height(v_uv + vec2(u_texel.x, 0.0)) - height(v_uv - vec2(u_texel.x, 0.0));
  float hy = height(v_uv + vec2(0.0, u_texel.y)) - height(v_uv - vec2(0.0, u_texel.y));
  vec3 n = normalize(vec3(-hx * 18.0, hy * 18.0, 1.0));   // y flipped: texture up, screen down
  vec2 p = vec2(v_uv.x, 1.0 - v_uv.y) * u_size;
  vec3 col = sky(p + n.xy * 90.0);

  vec3 light = normalize(vec3(-0.35, -0.75, 1.0));
  float spec = pow(max(dot(reflect(-light, n), vec3(0.0, 0.0, 1.0)), 0.0), 60.0);
  float flat_ = pow(max(dot(reflect(-light, vec3(0.0, 0.0, 1.0)), vec3(0.0, 0.0, 1.0)), 0.0), 60.0);
  float shade = dot(n, light) - light.z;              // slopes facing the light brighten, the others darken
  col *= 1.0 + shade * 0.9;
  col += vec3(0.95, 0.96, 0.97) * max(spec - flat_, 0.0) * 0.38;
  o = vec4(col, 1.0);
}`;

  function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  function program(gl, frag) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, frag));
    gl.bindAttribLocation(p, 0, "a_pos");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const uniforms = {};
    for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
      const name = gl.getActiveUniform(p, i).name;
      uniforms[name.replace("[0]", "")] = gl.getUniformLocation(p, name);
    }
    return { p, u: uniforms };
  }

  // "#rrggbb" custom property -> [r, g, b] in 0..1
  function color(name) {
    const hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim().replace("#", "");
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  }

  function water(hero) {
    if (hero.querySelector(".gn-water")) return;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const canvas = document.createElement("canvas");
    canvas.className = "gn-water";
    canvas.setAttribute("aria-hidden", "true");
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false });
    if (!gl || !gl.getExtension("EXT_color_buffer_float")) return;

    let step, drop, render;
    try {
      step = program(gl, STEP);
      drop = program(gl, DROP);
      render = program(gl, RENDER);
    } catch (e) {
      return;
    }

    const skyStops = new Float32Array([0, 1, 2, 3, 4, 5, 6].flatMap((i) => color(`--gn-sky-${i}`)));
    const sunStops = new Float32Array(["--gn-mist", "--gn-blue-300", "--gn-blue-400"].flatMap(color));

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    let W = 0, H = 0, simW = 0, simH = 0;
    let targets = [];

    function target(w, h) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return { tex, fb };
    }

    function resize() {
      const rect = hero.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = rect.width;
      H = rect.height;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      const scale = Math.min(1, 384 / W);
      const w = Math.max(32, Math.round(W * scale)), h = Math.max(32, Math.round(H * scale));
      if (w !== simW || h !== simH) {
        targets.forEach(({ tex, fb }) => (gl.deleteTexture(tex), gl.deleteFramebuffer(fb)));
        simW = w;
        simH = h;
        targets = [target(w, h), target(w, h)];
      }
      draw();
    }

    function pass(prog, setup) {
      const [src, dst] = targets;
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
      gl.viewport(0, 0, simW, simH);
      gl.useProgram(prog.p);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.uniform1i(prog.u.u_state, 0);
      setup(prog.u);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      targets = [dst, src];
    }

    function draw() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(render.p);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, targets[0].tex);
      gl.uniform1i(render.u.u_state, 0);
      gl.uniform2f(render.u.u_texel, 1 / simW, 1 / simH);
      gl.uniform2f(render.u.u_size, W, H);
      gl.uniform3fv(render.u.u_sky, skyStops);
      gl.uniform3fv(render.u.u_sun, sunStops);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    const drops = [];
    let last = null, frame = 0, quietUntil = 0;

    function addDrop(x, y, strength) {
      drops.push([x / W, 1 - y / H, strength]);
      quietUntil = performance.now() + 7000;
      if (!frame) frame = requestAnimationFrame(tick);
    }

    function tick(now) {
      frame = 0;
      if (!hero.isConnected) return;
      const radius = 22 / H;
      for (const [x, y, s] of drops.splice(0)) {
        pass(drop, (u) => {
          gl.uniform2f(u.u_center, x, y);
          gl.uniform1f(u.u_radius, radius);
          gl.uniform1f(u.u_strength, s);
          gl.uniform1f(u.u_aspect, W / H);
        });
      }
      for (let i = 0; i < 2; i++) {
        pass(step, (u) => {
          gl.uniform2f(u.u_texel, 1 / simW, 1 / simH);
          gl.uniform1f(u.u_damping, 0.988);
        });
      }
      draw();
      if (now < quietUntil) frame = requestAnimationFrame(tick);
    }

    hero.addEventListener("pointermove", (e) => {
      const rect = hero.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      if (last) {
        // a trail of drops along the path, stronger when the pointer moves faster
        const dx = x - last[0], dy = y - last[1], dist = Math.hypot(dx, dy);
        const n = Math.min(12, Math.ceil(dist / 10));
        const strength = Math.min(0.045, 0.012 + dist * 0.0012);
        for (let i = 1; i <= n; i++) addDrop(last[0] + (dx * i) / n, last[1] + (dy * i) / n, strength / Math.sqrt(n));
      } else {
        addDrop(x, y, 0.03);
      }
      last = [x, y];
    });
    hero.addEventListener("pointerleave", () => (last = null));
    hero.addEventListener("pointerdown", (e) => {
      const rect = hero.getBoundingClientRect();
      addDrop(e.clientX - rect.left, e.clientY - rect.top, 0.14);
    });

    hero.prepend(canvas);
    new ResizeObserver(resize).observe(hero);
    resize();
    hero.classList.add("gn-hero--water");
  }

  const init = () => document.querySelectorAll(".gn-hero").forEach(water);
  if (typeof document$ !== "undefined") document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
