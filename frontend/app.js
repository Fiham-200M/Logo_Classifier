/* ==========================================================================
   200M LOGO CLASSIFIER // APPLICATION LOGIC & DUAL-STATION CONTROLLER
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // Mode Switcher Elements
  const btnModeLogo = document.getElementById("btn-mode-logo");
  const btnModeFavicon = document.getElementById("btn-mode-favicon");
  const btnModeDual = document.getElementById("btn-mode-dual");
  const stationsWrapper = document.getElementById("upload-stations-wrapper");
  let activeStationMode = "logo"; // "logo" | "favicon" | "dual"

  // Logo Station Elements
  const logoDropZone = document.getElementById("logo-drop-zone");
  const logoFileInput = document.getElementById("logo-file-input");
  const btnBrowseLogo = document.getElementById("btn-browse-logo");
  const logoSampleChips = document.getElementById("logo-sample-chips");

  // Favicon Station Elements
  const favDropZone = document.getElementById("fav-drop-zone");
  const favFileInput = document.getElementById("fav-file-input");
  const btnBrowseFav = document.getElementById("btn-browse-fav");
  const favSampleChips = document.getElementById("fav-sample-chips");

  // Screenshot Station Elements
  const btnModeScreenshot = document.getElementById("btn-mode-screenshot");
  const screenshotDropZone = document.getElementById("screenshot-drop-zone");
  const screenshotFileInput = document.getElementById("screenshot-file-input");
  const btnBrowseScreenshot = document.getElementById("btn-browse-screenshot");

  // System & Loading Elements
  const loadingState = document.getElementById("loading-state");
  const loadingStage = document.getElementById("loading-stage");
  const loadingDetail = document.getElementById("loading-detail");
  const resultsDashboard = document.getElementById("results-dashboard");
  const systemStatus = document.getElementById("system-status");
  const statusLabel = document.getElementById("status-label");
  const protectedCount = document.getElementById("protected-count");

  // Verdict Elements
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictBadge = document.getElementById("verdict-badge");
  const threatBadge = document.getElementById("threat-badge");
  const mediaTypeBadge = document.getElementById("media-type-badge");
  const verifiedModeBadge = document.getElementById("verified-mode-badge");
  const matchedBrandName = document.getElementById("matched-brand-name");
  const verdictReason = document.getElementById("verdict-reason");
  const matchedVariantRow = document.getElementById("matched-variant-row");
  const matchedVariantName = document.getElementById("matched-variant-name");
  const confidenceNumber = document.getElementById("confidence-number");
  const gaugeBar = document.getElementById("gauge-bar");
  const latencyVal = document.getElementById("latency-val");

  // Micro-differences
  const microDiffsPanel = document.getElementById("micro-diffs-panel");
  const microDiffsList = document.getElementById("micro-diffs-list");

  // Visual Images
  const imgCandidate = document.getElementById("img-candidate");
  const imgReference = document.getElementById("img-reference");
  const imgHeatmap = document.getElementById("img-heatmap");
  const imgEdgeCand = document.getElementById("img-edge-cand");
  const imgEdgeRef = document.getElementById("img-edge-ref");
  const imgOcrVis = document.getElementById("img-ocr-vis");

  // Metrics
  const valSiglip = document.getElementById("val-siglip");
  const barSiglip = document.getElementById("bar-siglip");
  const valDinov2 = document.getElementById("val-dinov2");
  const barDinov2 = document.getElementById("bar-dinov2");
  const valDeltaE = document.getElementById("val-delta-e");
  const barDeltaE = document.getElementById("bar-delta-e");
  const detailDeltaE = document.getElementById("detail-delta-e");
  const valEdgeIou = document.getElementById("val-edge-iou");
  const barEdgeIou = document.getElementById("bar-edge-iou");
  const valOcrMatch = document.getElementById("val-ocr-match");
  const ocrCandText = document.getElementById("ocr-cand-text");
  const ocrRefText = document.getElementById("ocr-ref-text");
  const valClassifierConf = document.getElementById("val-classifier-conf");
  const valClassifierBrand = document.getElementById("val-classifier-brand");
  const valClassifierUnk = document.getElementById("val-classifier-unk");
  const shortlistTbody = document.getElementById("shortlist-tbody");

  // Brand Directory Modal
  const btnBrowseBrands = document.getElementById("btn-browse-brands");
  const brandsModal = document.getElementById("brands-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const brandSearchInput = document.getElementById("brand-search-input");
  const brandsModalGrid = document.getElementById("brands-modal-grid");
  let allBrandsData = [];

  // =========================================================================
  // 1. WORKSPACE MODE SWITCHER
  // =========================================================================
  function setWorkspaceMode(mode) {
    activeStationMode = mode;
    [btnModeLogo, btnModeFavicon, btnModeScreenshot, btnModeDual].forEach((btn) => btn && btn.classList.remove("active"));
    stationsWrapper.classList.remove("view-mode-logo", "view-mode-favicon", "view-mode-screenshot", "view-mode-dual");

    if (mode === "logo") {
      btnModeLogo.classList.add("active");
      stationsWrapper.classList.add("view-mode-logo");
    } else if (mode === "favicon") {
      btnModeFavicon.classList.add("active");
      stationsWrapper.classList.add("view-mode-favicon");
    } else if (mode === "screenshot") {
      if (btnModeScreenshot) btnModeScreenshot.classList.add("active");
      stationsWrapper.classList.add("view-mode-screenshot");
    } else {
      btnModeDual.classList.add("active");
      stationsWrapper.classList.add("view-mode-dual");
    }
  }

  btnModeLogo.addEventListener("click", () => setWorkspaceMode("logo"));
  btnModeFavicon.addEventListener("click", () => setWorkspaceMode("favicon"));
  if (btnModeScreenshot) btnModeScreenshot.addEventListener("click", () => setWorkspaceMode("screenshot"));
  btnModeDual.addEventListener("click", () => setWorkspaceMode("dual"));

  // =========================================================================
  // 2. HEALTH CHECK & SAMPLES LOADER
  // =========================================================================
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.status === "online") {
        statusLabel.textContent = `${data.brands_count} Brands Protected // ${data.device.toUpperCase()} Active`;
        protectedCount.textContent = data.brands_count;
        systemStatus.style.borderColor = "rgba(16, 185, 129, 0.4)";
      }
    } catch (err) {
      statusLabel.textContent = "Server Offline";
      systemStatus.style.borderColor = "rgba(239, 68, 68, 0.4)";
    }
  }

  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();

      // Render Logo presets
      logoSampleChips.innerHTML = "";
      const logoSamples = data.samples?.logos || (Array.isArray(data.samples) ? data.samples.filter(s => s.mode !== "favicon") : []);
      logoSamples.forEach((sample) => {
        const chip = document.createElement("div");
        chip.className = `sample-chip chip-${sample.category}`;
        chip.textContent = sample.label;
        chip.title = `Verify Logo: ${sample.path} (${sample.expected})`;
        chip.addEventListener("click", (e) => {
          e.stopPropagation();
          verifySample(sample.path, sample.label, "logo");
        });
        logoSampleChips.appendChild(chip);
      });

      // Render Favicon presets
      favSampleChips.innerHTML = "";
      const favSamples = data.samples?.favicons || (Array.isArray(data.samples) ? data.samples.filter(s => s.mode === "favicon") : []);
      favSamples.forEach((sample) => {
        const chip = document.createElement("div");
        chip.className = `sample-chip chip-${sample.category}`;
        chip.textContent = sample.label;
        chip.title = `Verify Favicon: ${sample.path} (${sample.expected})`;
        chip.addEventListener("click", (e) => {
          e.stopPropagation();
          verifySample(sample.path, sample.label, "favicon");
        });
        favSampleChips.appendChild(chip);
      });
    } catch (err) {
      logoSampleChips.innerHTML = '<span class="ribbon-hint">Failed to load samples</span>';
      favSampleChips.innerHTML = '<span class="ribbon-hint">Failed to load samples</span>';
    }
  }

  checkHealth();
  loadSamples();

  // =========================================================================
  // 3. STATION UPLOAD & DRAG-DROP HANDLERS
  // =========================================================================
  function setupDropZone(dropZone, fileInput, browseBtn, assetMode) {
    browseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    dropZone.addEventListener("click", (e) => {
      if (e.target !== browseBtn) fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileInput(e.dataTransfer.files[0], assetMode);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileInput(e.target.files[0], assetMode);
      }
    });
  }

  setupDropZone(logoDropZone, logoFileInput, btnBrowseLogo, "logo");
  setupDropZone(favDropZone, favFileInput, btnBrowseFav, "favicon");
  if (screenshotDropZone && screenshotFileInput && btnBrowseScreenshot) {
    setupDropZone(screenshotDropZone, screenshotFileInput, btnBrowseScreenshot, "screenshot");
  }

  // Clipboard Paste (Ctrl+V)
  window.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        const blob = items[i].getAsFile();
        let mode = "logo";
        if (activeStationMode === "favicon") mode = "favicon";
        else if (activeStationMode === "screenshot") mode = "screenshot";
        handleFileInput(blob, mode);
        break;
      }
    }
  });

  function handleFileInput(file, assetMode) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please upload a valid image file (PNG, JPG, WEBP, GIF, ICO).");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const b64 = e.target.result;
      verifyImage({ image_base64: b64, filename: file.name, asset_mode: assetMode }, b64, assetMode);
    };
    reader.readAsDataURL(file);
  }

  async function verifySample(samplePath, sampleLabel, assetMode) {
    const sampleImageUrl = `/api/sample_image?path=${encodeURIComponent(samplePath)}`;
    verifyImage({ sample_path: samplePath, filename: sampleLabel, asset_mode: assetMode }, sampleImageUrl, assetMode);
  }

  // =========================================================================
  // 4. API VERIFICATION PIPELINE
  // =========================================================================
  async function verifyImage(payload, displayImgUrl, assetMode) {
    loadingState.style.display = "block";
    resultsDashboard.style.display = "none";
    const isScreenshot = assetMode === "screenshot";
    const endpoint = isScreenshot ? "/api/verify_screenshot" : "/api/verify";

    loadingStage.textContent = isScreenshot
      ? "Scanning Webpage Screenshot for Protected Brands..."
      : `Running 200M ${assetMode === "favicon" ? "Favicon" : "Logo"} Multi-Modal Verification...`;

    loadingDetail.textContent = isScreenshot
      ? "Detecting text coordinates, clustering emblem regions, and executing forensic verification on GPU"
      : (assetMode === "favicon"
          ? "Evaluating micro-icon geometry consensus, color profile & multi-variant favicon signatures on CUDA"
          : "Extracting SigLIP 2, DINOv2 geometry, Canny edge strokes & CIE LAB color profiles on CUDA");

    const stages = isScreenshot
      ? [
          "Running fast OCR & layout parser across full screenshot...",
          "Locating brand text clusters and header proposals...",
          "Cropping candidate logo & emblem bounding boxes...",
          "Running SigLIP 2 & DINOv2 forensic verification on crops...",
          "Synthesizing detected regions and brand consensus...",
        ]
      : (assetMode === "favicon"
          ? [
              "Normalizing micro-icon geometry & generating dual canvases...",
              "Comparing candidate against 46 official brand favicons...",
              "Evaluating DINOv2 structural consensus & perceptual hashes...",
              "Verifying color palette stability and background transparency...",
              "Fusing forensic metrics in Decision Engine...",
            ]
          : [
              "Normalizing media & generating dual-contrast canvases...",
              "Extracting SigLIP 2 embeddings on CUDA...",
              "Evaluating DINOv2 structural geometry...",
              "Computing sub-pixel Canny edge strokes & IoU...",
              "Analyzing CIE LAB color profile & Delta E drift...",
              "Performing PaddleOCR text & token consensus...",
              "Fusing forensic metrics in Decision Engine...",
            ]);

    let stageIdx = 0;
    const stageTimer = setInterval(() => {
      stageIdx = (stageIdx + 1) % stages.length;
      loadingStage.textContent = stages[stageIdx];
    }, 400);

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      clearInterval(stageTimer);
      const data = await res.json();

      if (!data.success) {
        alert("Verification error: " + (data.error || "Unknown server error"));
        loadingState.style.display = "none";
        return;
      }

      renderResults(data, displayImgUrl, assetMode);
    } catch (err) {
      clearInterval(stageTimer);
      alert("Network or verification failure: " + err.message);
    } finally {
      loadingState.style.display = "none";
    }
  }

  // =========================================================================
  // 5. RENDER FORENSIC DASHBOARD
  // =========================================================================
  function renderResults(data, candidateImgSrc, assetMode) {
    const r = data.report;
    const vis = data.visualizations || {};
    const meta = r.preprocessing_meta || {};
    const metrics = r.forensic_metrics || {};

    // 1. Banner & Verdict
    verdictBanner.className = "verdict-banner glass-panel";
    verdictBadge.className = "verdict-badge";

    if (r.verdict === "MATCH") {
      verdictBanner.classList.add("verdict-match");
      verdictBadge.classList.add("badge-match");
    } else if (r.verdict === "REVIEW") {
      verdictBanner.classList.add("verdict-review");
      verdictBadge.classList.add("badge-review");
    } else {
      verdictBanner.classList.add("verdict-unknown");
      verdictBadge.classList.add("badge-unknown");
    }

    verdictBadge.textContent = r.verdict;
    threatBadge.textContent = r.threat_type || (assetMode === "screenshot" ? "PAGE_LOGO_DETECTED" : "EXACT_REPLICA");
    mediaTypeBadge.textContent = `${(r.media_type || assetMode).toUpperCase()} • ${meta.width || 0}x${meta.height || 0} ${meta.file_format || "PNG"}`;
    verifiedModeBadge.textContent = assetMode === "screenshot"
      ? "VERIFIED AS WEBPAGE SCREENSHOT"
      : (assetMode === "favicon" ? "VERIFIED AS FAVICON" : "VERIFIED AS BRAND LOGO");

    if (r.verdict !== "UNKNOWN") {
      matchedBrandName.textContent = r.brand_id;
      verdictTitleText(true, r.brand_id);
      matchedVariantRow.style.display = "inline-flex";
      matchedVariantName.textContent = r.matched_reference || "official_master.png";
    } else {
      matchedBrandName.textContent = "UNKNOWN / COMPETITOR";
      verdictTitleText(false, "");
      matchedVariantRow.style.display = "none";
    }

    verdictReason.textContent = r.action_reason || "Multi-modal vision analysis completed.";

    // Confidence Gauge
    const conf = Math.round((r.confidence_score || 0) * 100);
    confidenceNumber.textContent = `${conf}%`;
    const circumference = 2 * Math.PI * 42; // ~264
    const offset = circumference - (conf / 100) * circumference;
    gaugeBar.style.strokeDashoffset = offset;

    if (r.verdict === "MATCH") {
      gaugeBar.style.stroke = "var(--status-match)";
    } else if (r.verdict === "REVIEW") {
      gaugeBar.style.stroke = "var(--status-review)";
    } else {
      gaugeBar.style.stroke = "var(--status-unknown)";
    }

    latencyVal.textContent = `${r.processing_time_sec || 0.3}s`;

    // Micro-Differences
    const diffs = r.micro_differences_detected || [];
    if (diffs.length > 0 && r.verdict !== "MATCH") {
      microDiffsPanel.style.display = "flex";
      microDiffsList.innerHTML = diffs.map((d) => `<li>${escapeHtml(d)}</li>`).join("");
    } else {
      microDiffsPanel.style.display = "none";
    }

    // 2. Visual Comparator Images
    if (assetMode === "screenshot" && vis.alignment) {
      imgCandidate.src = vis.alignment;
    } else {
      imgCandidate.src = candidateImgSrc;
    }
    imgReference.src = vis.reference_image || candidateImgSrc;
    imgHeatmap.src = vis.diff_heatmap || "";
    imgEdgeCand.src = vis.edge_candidate || "";
    imgEdgeRef.src = vis.edge_reference || "";
    imgOcrVis.src = vis.ocr_vis || candidateImgSrc;

    // 3. Metrics Cards
    const siglipVal = metrics.siglip2_semantic_score ?? 0;
    valSiglip.textContent = siglipVal.toFixed(4);
    barSiglip.style.width = `${Math.min(100, Math.max(0, siglipVal * 100))}%`;

    const dinoVal = metrics.dinov2_geometry_score ?? 0;
    valDinov2.textContent = dinoVal.toFixed(4);
    barDinov2.style.width = `${Math.min(100, Math.max(0, dinoVal * 100))}%`;

    const dE = metrics.cielab_delta_e ?? 0;
    valDeltaE.textContent = dE.toFixed(2);
    const dEPercent = Math.min(100, (dE / 30) * 100);
    barDeltaE.style.width = `${dEPercent}%`;
    if (dE <= 5.0) {
      barDeltaE.style.background = "var(--status-match)";
      detailDeltaE.textContent = "ΔE ≤ 5.0 (Imperceptible difference)";
    } else if (dE <= 12.0) {
      barDeltaE.style.background = "var(--status-review)";
      detailDeltaE.textContent = "5.0 < ΔE ≤ 12.0 (Noticeable color drift)";
    } else {
      barDeltaE.style.background = "var(--status-unknown)";
      detailDeltaE.textContent = "ΔE > 12.0 (Significant color palette shift)";
    }

    const edgeIou = metrics.edge_stroke_iou ?? 0;
    valEdgeIou.textContent = edgeIou.toFixed(4);
    barEdgeIou.style.width = `${Math.min(100, Math.max(0, edgeIou * 100))}%`;

    const ocrMatch = metrics.ocr_text_match;
    valOcrMatch.textContent = ocrMatch ? "MATCH" : "NO MATCH";
    valOcrMatch.className = `metric-score badge-pill ${ocrMatch ? "badge-match" : "badge-review"}`;
    const candOcr = (r.ocr && (r.ocr.normalized_text || (r.ocr.detected_text && r.ocr.detected_text.join(" ")))) || data.report.ocr_evidence?.normalized_text || (r.detections && r.detections[0] && r.detections[0].text) || "";
    ocrCandText.textContent = `"${candOcr}"`;
    ocrRefText.textContent = `"${metrics.ref_ocr_text || r.brand_id || ""}"`;

    const cConf = (metrics.classifier_confidence || 0) * 100;
    valClassifierConf.textContent = `${cConf.toFixed(1)}%`;
    valClassifierBrand.textContent = metrics.classifier_predicted_brand || "unknown";
    valClassifierUnk.textContent = (metrics.classifier_unknown_prob || 0).toFixed(2);

    // 4. Candidate Shortlist Table
    const candidates = r.candidate_brands || [];
    shortlistTbody.innerHTML = candidates
      .map(
        (c, idx) => `
      <tr>
        <td>#${idx + 1}</td>
        <td><strong>${escapeHtml(c.brand_id)}</strong></td>
        <td>${c.siglip.toFixed(4)}</td>
        <td>${c.phash_dist}</td>
      </tr>
    `
      )
      .join("");

    // 5. Website Intelligence Profile (if available)
    const webIntelPanel = document.getElementById("website-intel-panel");
    const webIntel = r.website_intelligence || data.website_intelligence;
    if (webIntelPanel && webIntel && (webIntel.content || webIntel.domain)) {
      const content = webIntel.content || {};
      const primaryCat = content.primary || "Unknown";
      const catConf = Math.round((content.confidence || 0) * 100);
      const secondaries = content.secondary || [];
      const keywords = content.detected_keywords || [];

      document.getElementById("intel-primary-cat").textContent = primaryCat;
      document.getElementById("intel-cat-conf").textContent = `(${catConf}%)`;
      document.getElementById("intel-secondaries").textContent = secondaries.length ? `Subcategories: ${secondaries.join(", ")}` : "";
      
      const kwContainer = document.getElementById("intel-keywords");
      if (kwContainer) {
        kwContainer.innerHTML = keywords.slice(0, 12).map(k => `<span class="badge-pill badge-neutral" style="font-size: 0.72rem; padding: 2px 8px;">${escapeHtml(k)}</span>`).join("");
      }

      const textEv = webIntel.text_evidence || {};
      const visCtx = webIntel.visual_context || {};
      const signalsEl = document.getElementById("intel-signals");
      if (signalsEl) {
        let sigHtml = "";
        if (textEv.title) sigHtml += `<div><strong>Title:</strong> ${escapeHtml(textEv.title)}</div>`;
        if (textEv.headings && textEv.headings.length) sigHtml += `<div><strong>Heading:</strong> ${escapeHtml(textEv.headings[0])}</div>`;
        if (visCtx.raw_vlm_summary) sigHtml += `<div><strong>VLM:</strong> ${escapeHtml(visCtx.raw_vlm_summary)}</div>`;
        if (!sigHtml) sigHtml = `<div style="color: var(--text-muted);">Domain: ${escapeHtml(webIntel.domain || "N/A")} • No explicit text headers detected.</div>`;
        signalsEl.innerHTML = sigHtml;
      }
      webIntelPanel.style.display = "block";
    } else if (webIntelPanel) {
      webIntelPanel.style.display = "none";
    }

    resultsDashboard.style.display = "block";
    resultsDashboard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function verdictTitleText(isOfficial, brand) {
    const el = document.getElementById("verdict-title");
    if (isOfficial) {
      el.innerHTML = `Official Protected Brand: <span class="highlight">${escapeHtml(brand)}</span>`;
    } else {
      el.innerHTML = `<span class="highlight" style="color: var(--status-unknown);">Unrelated / Competitor Emblem</span>`;
    }
  }

  // =========================================================================
  // 6. VIEWER TABS
  // =========================================================================
  const tabs = document.querySelectorAll(".vtab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const target = tab.dataset.tab;
      document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));
      const activePanel = document.getElementById(`view-${target}`);
      if (activePanel) activePanel.classList.add("active");
    });
  });

  // =========================================================================
  // 7. BRAND DIRECTORY MODAL
  // =========================================================================
  btnBrowseBrands.addEventListener("click", async () => {
    brandsModal.style.display = "flex";
    if (allBrandsData.length === 0) {
      try {
        const res = await fetch("/api/brands");
        const data = await res.json();
        allBrandsData = data.brands || [];
        renderBrandsGrid(allBrandsData);
      } catch (e) {
        brandsModalGrid.innerHTML = "<p>Error loading brand catalog</p>";
      }
    }
  });

  btnCloseModal.addEventListener("click", () => {
    brandsModal.style.display = "none";
  });

  brandsModal.addEventListener("click", (e) => {
    if (e.target === brandsModal) brandsModal.style.display = "none";
  });

  brandSearchInput.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    const filtered = allBrandsData.filter((b) => b.brand_id.toLowerCase().includes(q));
    renderBrandsGrid(filtered);
  });

  function renderBrandsGrid(brands) {
    brandsModalGrid.innerHTML = brands
      .map(
        (b) => `
      <div class="brand-card-item">
        <span class="brand-card-name">${escapeHtml(b.brand_id)}</span>
        <span class="brand-card-vars">${b.variants_count} indexed variants</span>
      </div>
    `
      )
      .join("");
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
