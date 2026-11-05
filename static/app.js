/**
 * Object Detector & Counter - Modern Frontend Application
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    selectedFile: null,
    mode: 'standard', // 'standard' or 'open_vocab'
    cameraStream: null,
    facingMode: 'environment', // 'environment' or 'user'
  };

  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropZonePrompt = document.getElementById('drop-zone-prompt');
  const dropZonePreview = document.getElementById('drop-zone-preview');
  const selectedThumbnail = document.getElementById('selected-thumbnail');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');
  const btnRemoveFile = document.getElementById('btn-remove-file');

  const btnOpenCamera = document.getElementById('btn-open-camera');
  const cameraModal = document.getElementById('camera-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const cameraVideo = document.getElementById('camera-video');
  const cameraCanvas = document.getElementById('camera-canvas');
  const btnCapturePhoto = document.getElementById('btn-capture-photo');
  const btnSwitchCamera = document.getElementById('btn-switch-camera');

  const tabStandard = document.getElementById('tab-standard');
  const tabOpenVocab = document.getElementById('tab-openvocab');
  const openVocabContainer = document.getElementById('openvocab-container');
  const customLabelsInput = document.getElementById('custom-labels');
  const checkSlicing = document.getElementById('check-slicing');
  const presetDesk = document.getElementById('preset-desk');
  const presetStreet = document.getElementById('preset-street');

  const confidenceSlider = document.getElementById('confidence-slider');
  const confidenceVal = document.getElementById('confidence-val');
  const nmsSlider = document.getElementById('nms-slider');
  const nmsVal = document.getElementById('nms-val');

  const btnDetect = document.getElementById('btn-detect');
  const errorBanner = document.getElementById('error-banner');
  const errorMessage = document.getElementById('error-message');
  const btnCloseError = document.getElementById('btn-close-error');

  const loadingSpinner = document.getElementById('loading-spinner');
  const loadingStatus = document.getElementById('loading-status');
  const emptyState = document.getElementById('empty-state');
  const resultsContent = document.getElementById('results-content');
  const btnDownload = document.getElementById('btn-download');

  const metricTotal = document.getElementById('metric-total');
  const metricClasses = document.getElementById('metric-classes');
  const metricTime = document.getElementById('metric-time');
  const classChipsGrid = document.getElementById('class-chips-grid');
  const annotatedImage = document.getElementById('annotated-image');
  const detectionsTbody = document.getElementById('detections-tbody');

  const systemStatus = document.getElementById('system-status');
  const statusText = document.getElementById('status-text');
  const deviceBadge = document.getElementById('device-badge');

  // Palette generator for consistent class coloring in UI
  const PALETTE = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
    '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#A855F7'
  ];

  function getColorForClass(label) {
    let hash = 0;
    for (let i = 0; i < label.length; i++) {
      hash = label.charCodeAt(i) + ((hash << 5) - hash);
    }
    const idx = Math.abs(hash) % PALETTE.length;
    return PALETTE[idx];
  }

  // --- 1. Health Check & System Status ---
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        statusText.textContent = data.status === 'ok' ? 'Online' : 'Initializing';
        deviceBadge.textContent = data.device.toUpperCase();
        if (data.device.toLowerCase().includes('cuda')) {
          deviceBadge.style.color = '#10B981';
          deviceBadge.style.borderColor = 'rgba(16,185,129,0.3)';
        }
      } else {
        statusText.textContent = 'Server Unavailable';
        systemStatus.classList.remove('badge-pulse');
      }
    } catch {
      statusText.textContent = 'Offline';
      systemStatus.classList.remove('badge-pulse');
    }
  }
  checkHealth();

  // --- 2. Error Display Helpers ---
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
    errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideError() {
    errorBanner.classList.add('hidden');
    errorMessage.textContent = '';
  }

  btnCloseError.addEventListener('click', hideError);

  // --- 3. Drag and Drop Image Selection ---
  function handleSelectedFile(file) {
    hideError();
    if (!file) return;

    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
      showError('Please select a valid image format (JPG, PNG, or WEBP).');
      return;
    }

    // Validate 10 MB limit
    if (file.size > 10 * 1024 * 1024) {
      showError('Selected file exceeds the maximum 10 MB limit.');
      return;
    }

    state.selectedFile = file;

    // Read and show thumbnail
    const reader = new FileReader();
    reader.onload = (e) => {
      selectedThumbnail.src = e.target.result;
      previewFilename.textContent = file.name || 'captured_image.jpg';
      previewFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

      dropZonePrompt.classList.add('hidden');
      dropZonePreview.classList.remove('hidden');
      btnDetect.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearSelectedFile() {
    state.selectedFile = null;
    fileInput.value = '';
    selectedThumbnail.src = '';
    dropZonePreview.classList.add('hidden');
    dropZonePrompt.classList.remove('hidden');
    btnDetect.disabled = true;
  }

  dropZone.addEventListener('click', (e) => {
    if (e.target.closest('#btn-remove-file')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  // Drag & drop visual events
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleSelectedFile(dt.files[0]);
    }
  });

  // --- 4. Camera Capture Support ---
  async function startCamera() {
    try {
      hideError();
      if (state.cameraStream) {
        stopCamera();
      }
      const constraints = {
        video: {
          facingMode: state.facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };
      state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraVideo.srcObject = state.cameraStream;
      cameraModal.classList.remove('hidden');
    } catch (err) {
      showError(`Unable to access camera: ${err.message || 'Permission denied'}`);
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((track) => track.stop());
      state.cameraStream = null;
    }
    cameraModal.classList.add('hidden');
  }

  btnOpenCamera.addEventListener('click', startCamera);
  btnCloseModal.addEventListener('click', stopCamera);

  btnSwitchCamera.addEventListener('click', () => {
    state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
    startCamera();
  });

  btnCapturePhoto.addEventListener('click', () => {
    if (!state.cameraStream) return;

    cameraCanvas.width = cameraVideo.videoWidth || 640;
    cameraCanvas.height = cameraVideo.videoHeight || 480;
    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

    cameraCanvas.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], `camera_capture_${Date.now()}.jpg`, {
            type: 'image/jpeg',
          });
          handleSelectedFile(file);
          stopCamera();
        }
      },
      'image/jpeg',
      0.92
    );
  });

  // --- 5. Quick Sample Image Testing ---
  document.querySelectorAll('.sample-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const sampleName = btn.getAttribute('data-sample');
      try {
        hideError();
        const res = await fetch(`/static/sample_images/${sampleName}`);
        if (!res.ok) throw new Error('Failed to load sample image');
        const blob = await res.blob();
        const file = new File([blob], sampleName, { type: blob.type || 'image/jpeg' });
        handleSelectedFile(file);
      } catch (err) {
        showError(`Could not load sample image: ${err.message}`);
      }
    });
  });

  // --- 6. Detection Mode Switching & Presets ---
  tabStandard.addEventListener('click', () => {
    state.mode = 'standard';
    tabStandard.classList.add('active');
    tabStandard.setAttribute('aria-selected', 'true');
    tabOpenVocab.classList.remove('active');
    tabOpenVocab.setAttribute('aria-selected', 'false');
    openVocabContainer.classList.add('hidden');
    confidenceSlider.value = '0.35';
    confidenceVal.textContent = '0.35';
  });

  tabOpenVocab.addEventListener('click', () => {
    state.mode = 'open_vocab';
    tabOpenVocab.classList.add('active');
    tabOpenVocab.setAttribute('aria-selected', 'true');
    tabStandard.classList.remove('active');
    tabStandard.setAttribute('aria-selected', 'false');
    openVocabContainer.classList.remove('hidden');
    confidenceSlider.value = '0.15';
    confidenceVal.textContent = '0.15';
  });

  if (presetDesk) {
    presetDesk.addEventListener('click', () => {
      customLabelsInput.value = 'mobile phone, laptop charger, pen, laptop, book, backpack';
    });
  }

  if (presetStreet) {
    presetStreet.addEventListener('click', () => {
      customLabelsInput.value = 'car, person, bicycle, traffic light, bus, truck';
    });
  }

  // --- 7. Slider Realtime Value Displays ---
  confidenceSlider.addEventListener('input', (e) => {
    confidenceVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  nmsSlider.addEventListener('input', (e) => {
    nmsVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // --- 8. Run Object Detection ---
  btnDetect.addEventListener('click', async () => {
    if (!state.selectedFile) {
      showError('Please upload or capture an image first.');
      return;
    }

    hideError();
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    btnDetect.disabled = true;

    const startTime = performance.now();
    loadingStatus.textContent = state.mode === 'standard'
      ? 'Analyzing with RT-DETR...'
      : 'Querying OWLv2 Open-Vocabulary...';

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    formData.append('threshold', confidenceSlider.value);
    formData.append('nms_threshold', nmsSlider.value);
    formData.append('mode', state.mode);
    formData.append('enable_slicing', checkSlicing ? checkSlicing.checked : true);

    if (state.mode === 'open_vocab' && customLabelsInput.value.trim()) {
      formData.append('custom_labels', customLabelsInput.value.trim());
    }

    try {
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      const elapsedMs = Math.round(performance.now() - startTime);

      if (!response.ok) {
        throw new Error(data.detail || `Server returned error status ${response.status}`);
      }

      displayResults(data, elapsedMs);
    } catch (err) {
      showError(err.message || 'Detection failed. Please check backend server.');
      emptyState.classList.remove('hidden');
    } finally {
      loadingSpinner.classList.add('hidden');
      btnDetect.disabled = false;
    }
  });

  // --- 9. Display Detection Results ---
  function displayResults(data, elapsedMs) {
    // 1. Metric numbers
    metricTotal.textContent = data.total;
    const uniqueClassCount = Object.keys(data.counts || {}).length;
    metricClasses.textContent = uniqueClassCount;
    metricTime.textContent = `${elapsedMs} ms`;

    // 2. Class Chips
    classChipsGrid.innerHTML = '';
    if (uniqueClassCount === 0) {
      classChipsGrid.innerHTML = '<span class="text-secondary" style="font-size: 0.85rem;">No objects detected above the chosen confidence threshold. Try lowering the slider.</span>';
    } else {
      for (const [cls, count] of Object.entries(data.counts)) {
        const color = getColorForClass(cls);
        const chip = document.createElement('div');
        chip.className = 'class-chip';
        chip.innerHTML = `
          <span class="chip-dot" style="background-color: ${color}"></span>
          <span>${cls}</span>
          <span class="chip-count">${count}</span>
        `;
        classChipsGrid.appendChild(chip);
      }
    }

    // 3. Annotated Image
    annotatedImage.src = data.annotated_image;
    btnDownload.href = data.annotated_image;
    btnDownload.classList.remove('hidden');

    // 4. Detections Table
    detectionsTbody.innerHTML = '';
    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((det, idx) => {
        const row = document.createElement('tr');
        const color = getColorForClass(det.label);
        const pct = Math.round(det.confidence * 100);
        const boxStr = `[${det.box.map((n) => Math.round(n)).join(', ')}]`;

        row.innerHTML = `
          <td>${idx + 1}</td>
          <td>
            <span class="label-badge">
              <span class="badge-indicator" style="background-color: ${color}"></span>
              ${det.label}
            </span>
          </td>
          <td>
            <div class="confidence-cell">
              <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: ${pct}%;"></div>
              </div>
              <span class="confidence-pct">${pct}%</span>
            </div>
          </td>
          <td class="box-coords">${boxStr}</td>
        `;
        detectionsTbody.appendChild(row);
      });
    } else {
      detectionsTbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No object detections</td></tr>';
    }

    resultsContent.classList.remove('hidden');
    resultsContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});

