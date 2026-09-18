(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtMs = (ms) => ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : (ms >= 10 ? Math.round(ms) : ms.toFixed(1)) + " ms";
  const fmtBytes = (b) => b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : b >= 1024 ? Math.round(b / 1024) + " kB" : b + " B";

  class ApiError extends Error {
    constructor(status, code, message) { super(message); this.status = status; this.code = code; }
  }
  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new ApiError(r.status, body.code || "http_error", body.message || r.statusText);
    return body;
  }

  // ---------------------------------------------------------------- scenarios
  const EXAMPLE = {
    id: "triage", title: "Payment triage (3 fields)",
    description: "The request fixture used by the tests and the benchmark.",
    context: "Customer requested an unusual wire transfer of 48,000 EUR to a newly added beneficiary in another country, minutes after changing their contact email and phone number from an unrecognised device. The customer's normal activity is small domestic card payments.",
    fields: [
      { name: "priority", description: "Required handling priority for the support team", choices: ["LOW", "MEDIUM", "HIGH"] },
      { name: "fraudulent", description: "Whether fraud is suspected", choices: [false, true] },
      { name: "tier", description: "Escalation tier for the risk desk", choices: ["TIER_1_LOW", "TIER_2_MEDIUM", "TIER_3_HIGH"] },
    ],
  };

  let presets = [EXAMPLE];
  let current = EXAMPLE;
  let sampleIdx = -1;

  // ---------------------------------------------------------------- views
  function showView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("is-active", v.id === "view-" + name));
    document.querySelectorAll(".view-tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === name));
    if (name !== "tetris" && window.pcdTetris) window.pcdTetris.pause();
    const params = new URLSearchParams(location.search);
    if (name === "demo") params.delete("view"); else params.set("view", name);
    const query = params.toString();
    history.replaceState(null, "", location.pathname + (query ? "?" + query : ""));
  }
  document.querySelectorAll(".view-tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

  // ---------------------------------------------------------------- fields editor
  function addRow(field = { name: "", description: "", choices: [] }) {
    const boolean = field.choices.length && typeof field.choices[0] === "boolean";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input name="name" placeholder="field_name" value="${esc(field.name)}" required></td>
      <td><button type="button" class="type-toggle" data-type="${boolean ? "boolean" : "enum"}" aria-label="Field type, click to switch">${boolean ? "boolean" : "choice"}</button></td>
      <td><input name="description" placeholder="What this field decides" value="${esc(field.description)}"></td>
      <td><input name="choices" placeholder="A, B, C" value="${esc(boolean ? "" : field.choices.join(", "))}"${boolean ? " disabled" : ""}></td>
      <td><button type="button" class="remove" aria-label="Remove field">×</button></td>`;
    tr.querySelector(".remove").addEventListener("click", () => tr.remove());
    tr.querySelector(".type-toggle").addEventListener("click", (e) => {
      const boolean = e.currentTarget.dataset.type !== "boolean";
      e.currentTarget.dataset.type = boolean ? "boolean" : "enum";
      e.currentTarget.textContent = boolean ? "boolean" : "choice";
      tr.querySelector('input[name="choices"]').disabled = boolean;
    });
    $("field-rows").appendChild(tr);
  }
  $("field-add").addEventListener("click", () => addRow());

  function updateFieldCount() {
    const n = $("field-rows").querySelectorAll("tr").length;
    $("field-count").textContent = n ? `(${n})` : "";
  }
  $("field-rows").addEventListener("click", () => setTimeout(updateFieldCount, 0));
  $("field-add").addEventListener("click", updateFieldCount);

  /** Shows sample i of the current scenario in the textarea (scenarios without samples show their context). */
  function showSample(i) {
    const samples = current.samples || [];
    sampleIdx = samples.length ? ((i % samples.length) + samples.length) % samples.length : -1;
    const s = sampleIdx >= 0 ? samples[sampleIdx] : null;
    $("context").value = s ? s.context : current.context;
    $("sample-label").textContent = s ? `${sampleIdx + 1} of ${samples.length}: ${s.label}` : "";
    const expected = s && s.expected ? Object.entries(s.expected) : [];
    $("expected-note").hidden = !expected.length;
    $("expected-note").textContent = expected.length
      ? "Filed as " + expected.map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(", ") + "."
      : "";
    $("next-sample").hidden = samples.length < 2;
  }
  $("next-sample").addEventListener("click", () => showSample(sampleIdx + 1));

  function loadPreset(id) {
    current = presets.find((p) => p.id === id) || presets[0];
    $("preset-select").value = current.id;
    $("preset-hint").textContent = current.description || "Edit the input and the fields, then decode.";
    $("field-rows").innerHTML = "";
    current.fields.forEach(addRow);
    updateFieldCount();
    showSample(0);
    hideProblem();
  }
  $("preset-select").addEventListener("change", (e) => loadPreset(e.target.value));

  async function loadPresets() {
    try {
      const more = await api("/presets.json");
      presets = [EXAMPLE, ...more];
    } catch (e) {
      presets = [EXAMPLE];
    }
    $("preset-select").innerHTML = presets.map((p) => `<option value="${esc(p.id)}">${esc(p.title)}</option>`).join("");
    const wanted = new URLSearchParams(location.search).get("preset");
    loadPreset(presets.some((p) => p.id === wanted) ? wanted : EXAMPLE.id);
  }

  function readRequest() {
    const fields = [...$("field-rows").querySelectorAll("tr")].map((tr) => {
      const get = (n) => tr.querySelector(`[name="${n}"]`).value.trim();
      const boolean = tr.querySelector(".type-toggle").dataset.type === "boolean";
      return {
        name: get("name"),
        description: get("description"),
        choices: boolean ? [false, true] : get("choices").split(",").map((s) => s.trim()).filter(Boolean),
      };
    });
    return { context: $("context").value, fields };
  }

  // ---------------------------------------------------------------- model
  async function refreshModels() {
    const status = $("model-status");
    try {
      const [health, models] = await Promise.all([api("/health"), api("/v1/models")]);
      const sel = $("model-select");
      sel.innerHTML = models.models.map((m) => `<option value="${esc(m.id)}"${m.active ? " selected" : ""}>${esc(m.id)} (${fmtBytes(m.bytes)})</option>`).join("")
        || `<option value="">no .gguf files in the models directory</option>`;
      $("model-switch").disabled = !models.models.length;
      status.textContent = health.ready ? `${health.description} on ${health.backend}` : "no model loaded";
      status.className = "status " + (health.ready ? "ok" : "bad");
    } catch (e) {
      status.textContent = "server unreachable";
      status.className = "status bad";
    }
  }
  $("model-switch").addEventListener("click", async () => {
    const id = $("model-select").value;
    if (!id) return;
    const btn = $("model-switch");
    btn.disabled = true; btn.textContent = "Loading…";
    hideProblem();
    try {
      await api("/v1/models/select", { method: "POST", body: JSON.stringify({ id }) });
    } catch (e) {
      showProblem(e);
    } finally {
      btn.textContent = "Switch"; btn.disabled = false;
      refreshModels();
    }
  });

  // ---------------------------------------------------------------- problems
  function showProblem(e) {
    const el = $("problem");
    el.hidden = false;
    el.innerHTML = e instanceof ApiError
      ? `<code>${e.status} ${esc(e.code)}</code> ${esc(e.message)}`
      : `Could not reach the server: ${esc(e.message)}`;
  }
  function hideProblem() { $("problem").hidden = true; }

  // ---------------------------------------------------------------- result
  const PHASES = [
    ["tokenize", "tokenize", "tokenize"],
    ["restoreOrPrefill", "restore checkpoint", "restore"],
    ["dynamicContext", "dynamic context", "dynamic"],
    ["broadcast", "broadcast", "broadcast"],
    ["suffix", "field suffixes", "suffix"],
    ["tree", "collision tree", "tree"],
  ];

  function render(res) {
    const m = res.metrics;
    $("s-ms").textContent = fmtMs(m.elapsedMs);
    $("s-passes").textContent = m.forwardPasses;
    const cache = $("s-cache");
    cache.textContent = m.schemaCacheStatus === "hit" ? "hit (restored)" : m.schemaCacheStatus === "miss" ? "miss (prefilled)" : "fallback";
    cache.className = m.schemaCacheStatus;
    $("s-ckpt").textContent = m.checkpointBytes ? fmtBytes(m.checkpointBytes) : "–";

    const list = $("result-fields");
    list.className = "fields";
    list.innerHTML = res.fields.map((f) => {
      const choices = Object.entries(f.probabilities);
      const win = JSON.stringify(f.value);
      return `<li>
        <div class="fhead"><span class="fname">${esc(f.name)}</span><span class="fval">${esc(win)}</span><span class="flevels">${f.levels} level${f.levels === 1 ? "" : "s"}</span></div>
        ${choices.map(([label, p]) => `<div class="choice${label === String(f.value) ? " winner" : ""}">
          <span class="clabel" title="${esc(label)}">${esc(label)}</span>
          <span class="cbar"><i style="width:${(p * 100).toFixed(2)}%"></i></span>
          <span class="cprob">${(p * 100).toFixed(1)}%</span>
        </div>`).join("")}
      </li>`;
    }).join("");

    $("values-json").textContent = JSON.stringify(res.values, null, 2);
    $("copy-values").hidden = false;

    const total = PHASES.reduce((s, [k]) => s + (m.phasesMs[k] || 0), 0) || 1;
    const restoreClass = m.schemaCacheStatus === "hit" ? "restore" : "prefill";
    $("phases").innerHTML = PHASES.map(([k, , cls]) => {
      const ms = m.phasesMs[k] || 0;
      const c = k === "restoreOrPrefill" ? restoreClass : cls;
      return `<i class="${c}" style="width:${(ms / total * 100).toFixed(2)}%" title="${esc(k)} ${fmtMs(ms)}"></i>`;
    }).join("");
    $("legend").innerHTML = PHASES.map(([k, name, cls]) => {
      const c = k === "restoreOrPrefill" ? restoreClass : cls;
      const label = k === "restoreOrPrefill" ? (m.schemaCacheStatus === "hit" ? "restore checkpoint" : "prefill schema prefix") : name;
      return `<li><span class="lname"><b class="${c}"></b>${esc(label)}</span><span class="lms">${fmtMs(m.phasesMs[k] || 0)}</span></li>`;
    }).join("");
    $("phases-note").textContent = m.schemaCacheStatus === "hit"
      ? "The schema prefix came from a checkpoint; only your input and the field openers were decoded."
      : m.schemaCacheStatus === "miss"
        ? "First run of this schema: the prefix was prefilled and checkpointed. Run again to see the restore."
        : "The cached checkpoint could not be restored, so the prefix was recomputed. The bad checkpoint was dropped.";
  }

  function showRequest(request) {
    $("request-json").textContent = JSON.stringify(request, null, 2);
    $("copy-curl").hidden = false;
  }

  // ---------------------------------------------------------------- decode
  async function decode() {
    const btn = $("decode");
    const request = readRequest();
    showRequest(request);
    hideProblem();
    btn.disabled = true; btn.textContent = "Decoding…";
    $("result-fields").classList.add("pending");
    try {
      render(await api("/v1/pcd/decode", { method: "POST", body: JSON.stringify(request) }));
    } catch (e) {
      showProblem(e);
    } finally {
      btn.disabled = false; btn.textContent = "Decode";
      $("result-fields").classList.remove("pending");
    }
  }
  $("decode").addEventListener("click", decode);
  $("context").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") decode(); });

  $("copy-values").addEventListener("click", () => navigator.clipboard.writeText($("values-json").textContent));
  $("copy-curl").addEventListener("click", () => {
    const body = JSON.stringify(readRequest());
    navigator.clipboard.writeText(`curl -s -X POST ${location.origin}/v1/pcd/decode -H 'Content-Type: application/json' -d '${body.replace(/'/g, "'\\''")}'`);
  });

  loadPresets();
  refreshModels();
  const view = new URLSearchParams(location.search).get("view");
  if (view === "tetris") showView("tetris");
})();
