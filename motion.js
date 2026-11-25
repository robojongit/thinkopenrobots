/* Decorative motion. Plain JS, no dependencies.
 * Nothing here gates content: without this file, or with reduced motion,
 * every element simply shows its final, static state. */
(function () {
  "use strict";

  var reduceQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  var enabled = "IntersectionObserver" in window && !(reduceQuery && reduceQuery.matches);
  var root = document.documentElement;
  if (enabled) root.classList.add("motion");

  /* ---------- one-time reveals ---------- */

  var revealer = enabled ? new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      en.target.classList.add("revealed");
      revealer.unobserve(en.target);
    });
  }, { rootMargin: "0px 0px -12% 0px" }) : null;

  function reveal(node) {
    if (!node || node.classList.contains("revealed")) return;
    if (!revealer) { node.classList.add("revealed"); return; }
    revealer.observe(node);
  }
  window.SiteMotion = { reveal: reveal };

  /* ---------- loops that run only while visible ---------- */

  function whileVisible(node, onChange) {
    var inView = false;
    function update() { onChange(inView && !document.hidden); }
    new IntersectionObserver(function (entries) {
      inView = entries[entries.length - 1].isIntersecting;
      update();
    }).observe(node);
    document.addEventListener("visibilitychange", update);
  }

  /* ---------- hero arm: reach, grip, lift, carry, place, release, return ---------- */

  function armLoop(svg) {
    var S = [124, 340], ED = [222, 168], WD = [380, 196];
    var L1 = Math.hypot(ED[0] - S[0], ED[1] - S[1]);
    var L2 = Math.hypot(WD[0] - ED[0], WD[1] - ED[1]);
    var A1D = Math.atan2(ED[1] - S[1], ED[0] - S[0]);
    var A2D = Math.atan2(WD[1] - ED[1], WD[0] - ED[0]);
    var DEG = 180 / Math.PI, OPEN = 5;

    var ups = svg.querySelectorAll(".arm-up"), fores = svg.querySelectorAll(".arm-fore");
    var hand = svg.querySelector(".arm-hand"), cube = svg.querySelector(".arm-cube"), guides = svg.querySelector(".arm-guides");
    var fl = svg.querySelector(".arm-finger-l"), fr = svg.querySelector(".arm-finger-r");

    /* Wrist targets. D is the drawn pose (red cube held over the green one);
     * the cube's offset from its drawn spot: held = W - D, placed (0, 44), ground (-118, 104). */
    var D = [380, 196], P = [380, 240], R = [380, 186], H = [300, 150], A = [262, 226], G = [262, 300];
    var ON = [0, 44], GND = [-118, 104], LIFT = [A[0] - D[0], A[1] - D[1]];
    /* time (s), wrist, gripper open 0..1, guides opacity, cube offset, cube opacity */
    var K = [
      [0.0, D, 0, 1, [0, 0], 1],
      [0.7, D, 0, 1, [0, 0], 1],
      [1.5, P, 0, 0, ON, 1],
      [1.9, P, 1, 0, ON, 1],
      [2.5, R, 1, 0, ON, 1],
      [3.7, H, 1, 0, ON, 1],
      [4.1, H, 1, 0, ON, 0],
      [4.2, H, 1, 0, GND, 0],
      [4.6, H, 1, 0, GND, 1],
      [5.6, A, 1, 0, GND, 1],
      [6.2, G, 1, 0, GND, 1],
      [6.6, G, 0, 0, GND, 1],
      [7.2, A, 0, 0, LIFT, 1],
      [8.5, D, 0, 0, [0, 0], 1],
      [8.9, D, 0, 1, [0, 0], 1],
      [9.6, D, 0, 1, [0, 0], 1]
    ];
    var PERIOD = K[K.length - 1][0];

    function ease(x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; }
    function mix(a, b, k) { return a + (b - a) * k; }

    function pose(t) {
      var i = 0;
      while (i < K.length - 2 && t >= K[i + 1][0]) i++;
      var a = K[i], b = K[i + 1];
      var k = ease(Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0]))));
      return {
        w: [mix(a[1][0], b[1][0], k), mix(a[1][1], b[1][1], k)],
        open: mix(a[2], b[2], k), guides: mix(a[3], b[3], k),
        cube: [mix(a[4][0], b[4][0], k), mix(a[4][1], b[4][1], k)], cubeOp: mix(a[5], b[5], k)
      };
    }

    function setAll(list, value) {
      for (var i = 0; i < list.length; i++) list[i].setAttribute("transform", value);
    }
    function render(p) {
      var dx = p.w[0] - S[0], dy = p.w[1] - S[1];
      var d = Math.min(Math.max(Math.hypot(dx, dy), Math.abs(L1 - L2) + 1), L1 + L2 - 1);
      var cosA = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d);
      var a1 = Math.atan2(dy, dx) - Math.acos(Math.max(-1, Math.min(1, cosA)));  /* elbow up */
      var e = [S[0] + L1 * Math.cos(a1), S[1] + L1 * Math.sin(a1)];
      var a2 = Math.atan2(p.w[1] - e[1], p.w[0] - e[0]);
      setAll(ups, "rotate(" + ((a1 - A1D) * DEG).toFixed(3) + " " + S[0] + " " + S[1] + ")");
      setAll(fores, "translate(" + (e[0] - ED[0]).toFixed(2) + " " + (e[1] - ED[1]).toFixed(2) + ") rotate(" +
        ((a2 - A2D) * DEG).toFixed(3) + " " + ED[0] + " " + ED[1] + ")");
      hand.setAttribute("transform", "translate(" + (p.w[0] - WD[0]).toFixed(2) + " " + (p.w[1] - WD[1]).toFixed(2) + ")");
      fl.setAttribute("transform", "translate(" + (-OPEN * p.open).toFixed(2) + " 0)");
      fr.setAttribute("transform", "translate(" + (OPEN * p.open).toFixed(2) + " 0)");
      cube.setAttribute("transform", "translate(" + p.cube[0].toFixed(2) + " " + p.cube[1].toFixed(2) + ")");
      cube.setAttribute("opacity", p.cubeOp.toFixed(3));
      guides.setAttribute("opacity", p.guides.toFixed(3));
    }
    function reset() {
      [].concat(Array.prototype.slice.call(ups), Array.prototype.slice.call(fores), [hand, fl, fr, cube])
        .forEach(function (n) { n.removeAttribute("transform"); });
      cube.removeAttribute("opacity");
      guides.removeAttribute("opacity");
    }

    var elapsed = 0, last = null, frame = 0, running = false;
    function tick(ts) {
      frame = 0;
      if (!running) return;
      if (last !== null) elapsed = (elapsed + Math.min(ts - last, 64) / 1000) % PERIOD;
      last = ts;
      render(pose(elapsed));
      frame = requestAnimationFrame(tick);
    }
    function setRunning(on) {
      running = on && !(reduceQuery && reduceQuery.matches);
      if (running && !frame) { last = null; frame = requestAnimationFrame(tick); }
    }
    whileVisible(svg, setRunning);
    if (reduceQuery && reduceQuery.addEventListener) {
      reduceQuery.addEventListener("change", function (q) {
        if (q.matches) { running = false; reset(); }
      });
    }
  }

  function boot() {
    if (!enabled) return;
    var arm = document.getElementById("arm");
    if (arm) armLoop(arm);
    Array.prototype.forEach.call(document.querySelectorAll("[data-loop]"), function (node) {
      whileVisible(node, function (on) { node.classList.toggle("playing", on); });
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-reveal]"), reveal);
  }
  boot();
})();
