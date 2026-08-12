(function () {
  "use strict";

  // ---------- Language (mirrors the parent portfolio site's i18n.js) ----------
  var root = document.documentElement;
  var LANG_KEY = "inferlab-lang";

  function applyLang(lang) {
    if (lang === "ja") {
      root.setAttribute("data-lang", "ja");
      root.setAttribute("lang", "ja");
    } else {
      root.removeAttribute("data-lang");
      root.setAttribute("lang", "en");
    }
  }
  function isJa() { return root.getAttribute("data-lang") === "ja"; }

  var savedLang = localStorage.getItem(LANG_KEY);
  if (!savedLang) {
    savedLang = navigator.language && navigator.language.toLowerCase().indexOf("ja") === 0 ? "ja" : "en";
  }
  applyLang(savedLang);

  document.getElementById("langToggle").addEventListener("click", function () {
    var next = isJa() ? "en" : "ja";
    applyLang(next);
    localStorage.setItem(LANG_KEY, next);
    if (lastBenchResults) renderBenchmark(lastBenchResults);
  });

  function t(en, ja) { return isJa() ? ja : en; }

  // ---------- DOM refs ----------
  var promptEl = document.getElementById("prompt");
  var maxTokensEl = document.getElementById("maxTokens");
  var valMaxTokens = document.getElementById("valMaxTokens");
  var quantizeEl = document.getElementById("quantize");
  var genBtn = document.getElementById("genBtn");
  var benchBtn = document.getElementById("benchBtn");
  var outputEl = document.getElementById("output");
  var genStatus = document.getElementById("genStatus");
  var genStats = document.getElementById("genStats");
  var benchBarsEl = document.getElementById("benchBars");
  var memBarsEl = document.getElementById("memBars");

  maxTokensEl.addEventListener("input", function () {
    valMaxTokens.textContent = maxTokensEl.value;
  });

  var lastBenchResults = null;

  // ---------- Generate (SSE-over-fetch streaming) ----------
  function streamGenerate(prompt, maxTokens, quantize, onToken, onDone, onError) {
    fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt, maxTokens: maxTokens, quantize: quantize })
    }).then(function (resp) {
      if (!resp.ok || !resp.body) {
        onError(new Error("HTTP " + resp.status));
        return;
      }
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buf = "";

      function pump() {
        return reader.read().then(function (result) {
          if (result.done) {
            onDone();
            return;
          }
          buf += decoder.decode(result.value, { stream: true });
          var chunks = buf.split("\n\n");
          buf = chunks.pop();
          for (var i = 0; i < chunks.length; i++) {
            var chunk = chunks[i];
            var eventMatch = /^event: (.*)$/m.exec(chunk);
            var dataMatch = /^data: ([\s\S]*)$/m.exec(chunk);
            var event = eventMatch ? eventMatch[1] : "message";
            var data = dataMatch ? dataMatch[1] : "";
            if (event === "token") onToken(data);
            else if (event === "done") { onDone(); return; }
          }
          return pump();
        });
      }
      return pump();
    }).catch(onError);
  }

  genBtn.addEventListener("click", function () {
    var prompt = promptEl.value || "Once upon a time";
    var maxTokens = parseInt(maxTokensEl.value, 10);
    var quantize = quantizeEl.checked;

    genBtn.disabled = true;
    outputEl.textContent = "";
    var cursor = document.createElement("span");
    cursor.className = "cursor";
    outputEl.appendChild(cursor);
    genStatus.textContent = t("Generating…", "生成中…");
    genStats.textContent = "";

    var tokenCount = 0;
    var start = performance.now();

    streamGenerate(prompt, maxTokens, quantize, function onToken(piece) {
      tokenCount++;
      outputEl.insertBefore(document.createTextNode(piece), cursor);
      var elapsed = (performance.now() - start) / 1000;
      genStats.textContent = tokenCount + " " + t("tokens", "トークン") + " · " +
        (elapsed > 0 ? (tokenCount / elapsed).toFixed(1) : "0") + " tok/s";
    }, function onDone() {
      cursor.remove();
      genBtn.disabled = false;
      genStatus.textContent = t("Done.", "完了。");
    }, function onError(err) {
      cursor.remove();
      genBtn.disabled = false;
      genStatus.textContent = t("Error: ", "エラー: ") + err.message;
    });
  });

  // ---------- Benchmark ----------
  var LEG_LABELS = {
    no_cache_fp32: { en: "No cache (fp32)", ja: "キャッシュなし (fp32)" },
    cached_fp32: { en: "KV cache (fp32)", ja: "KVキャッシュ (fp32)" },
    cached_int8: { en: "KV cache + int8", ja: "KVキャッシュ + int8" },
    sequential_x3_fp32: { en: "3 sequential (fp32)", ja: "逐次3件 (fp32)" },
    batched_x3_fp32: { en: "3 batched (fp32)", ja: "バッチ3件 (fp32)" }
  };

  function renderBenchmark(results) {
    benchBarsEl.innerHTML = "";
    var maxTps = 0;
    results.forEach(function (r) { if (r.tokensPerSec > maxTps) maxTps = r.tokensPerSec; });

    results.forEach(function (r) {
      var label = LEG_LABELS[r.label] || { en: r.label, ja: r.label };
      var row = document.createElement("div");
      row.className = "bench-row";

      var labelRow = document.createElement("div");
      labelRow.className = "bench-label";
      var name = document.createElement("span");
      name.textContent = t(label.en, label.ja);
      var num = document.createElement("span");
      num.className = "num";
      num.textContent = r.tokensPerSec.toFixed(1) + " tok/s";
      labelRow.appendChild(name);
      labelRow.appendChild(num);

      var track = document.createElement("div");
      track.className = "bar-track";
      var fill = document.createElement("div");
      fill.className = "bar-fill" + (r.label.indexOf("int8") >= 0 || r.label.indexOf("batched") >= 0 ? " good" : "");
      var pct = maxTps > 0 ? Math.max(2, (r.tokensPerSec / maxTps) * 100) : 0;
      fill.style.width = pct + "%";
      track.appendChild(fill);

      row.appendChild(labelRow);
      row.appendChild(track);
      benchBarsEl.appendChild(row);
    });

    memBarsEl.innerHTML = "";
    var fp32 = results.filter(function (r) { return r.label === "cached_fp32"; })[0];
    var int8 = results.filter(function (r) { return r.label === "cached_int8"; })[0];
    if (fp32 && int8) {
      var maxBytes = Math.max(fp32.memoryBytes, int8.memoryBytes);
      [["fp32", fp32.memoryBytes, ""], ["int8", int8.memoryBytes, " good"]].forEach(function (entry) {
        var col = document.createElement("div");
        col.className = "mem-col";
        var labelRow = document.createElement("div");
        labelRow.className = "bench-label";
        var name = document.createElement("span");
        name.textContent = entry[0];
        var num = document.createElement("span");
        num.className = "num";
        num.textContent = (entry[1] / 1e6).toFixed(1) + " MB";
        labelRow.appendChild(name);
        labelRow.appendChild(num);
        var track = document.createElement("div");
        track.className = "bar-track";
        var fill = document.createElement("div");
        fill.className = "bar-fill" + entry[2];
        fill.style.width = Math.max(2, (entry[1] / maxBytes) * 100) + "%";
        track.appendChild(fill);
        col.appendChild(labelRow);
        col.appendChild(track);
        memBarsEl.appendChild(col);
      });
    }
  }

  benchBtn.addEventListener("click", function () {
    var prompt = promptEl.value || "Once upon a time";
    var maxTokens = parseInt(maxTokensEl.value, 10);

    benchBtn.disabled = true;
    benchBarsEl.innerHTML = "<p class=\"hint\" style=\"margin-top:0;\">" + t("Running…", "実行中…") + "</p>";

    fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt, maxTokens: maxTokens })
    }).then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return resp.json();
    }).then(function (data) {
      lastBenchResults = data.results || [];
      renderBenchmark(lastBenchResults);
    }).catch(function (err) {
      benchBarsEl.innerHTML = "<p class=\"hint\" style=\"margin-top:0;color:var(--bad);\">" + t("Error: ", "エラー: ") + err.message + "</p>";
    }).finally(function () {
      benchBtn.disabled = false;
    });
  });
})();
