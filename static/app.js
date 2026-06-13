document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let selectedFile = null;
    let accuracyChart = null;
    let parametersChart = null;
    let latencyChart = null;

    // --- DOM ELEMENTS ---
    const fileInput = document.getElementById('file-input');
    const fileDropzone = document.getElementById('file-dropzone');
    const dropzoneDefault = document.getElementById('dropzone-default');
    const dropzoneActive = document.getElementById('dropzone-active');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const filePreviewIcon = document.getElementById('file-preview-icon');
    
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnReset = document.getElementById('btn-reset');
    
    // Tab controls
    const tabUpload = document.getElementById('tab-upload');
    const tabUrl = document.getElementById('tab-url');
    const uploadContainer = document.getElementById('upload-container');
    const urlContainer = document.getElementById('url-container');
    const videoUrlInput = document.getElementById('video-url-input');
    const btnScanUrl = document.getElementById('btn-scan-url');
    
    const analysisStatus = document.getElementById('analysis-status');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const resultsArea = document.getElementById('results-area');
    const emptyStateResults = document.getElementById('empty-state-results');
    const resultsContent = document.getElementById('results-content');
    
    // Status Steps
    const stepUpload = document.getElementById('step-upload');
    const stepFace = document.getElementById('step-face');
    const stepSpatial = document.getElementById('step-spatial');
    const stepTemporal = document.getElementById('step-temporal');

    // Results elements
    const croppedFaceImg = document.getElementById('cropped-face-img');
    const consensusBadge = document.getElementById('consensus-badge');
    const consensusDetails = document.getElementById('consensus-details');
    const overallVerdictBox = document.getElementById('overall-verdict-box');
    
    // EfficientNet display elements
    const scoreEff = document.getElementById('score-eff');
    const labelEff = document.getElementById('label-eff');
    const circleEff = document.getElementById('circle-eff');
    const heatmapEff = document.getElementById('heatmap-eff');
    const latencyEff = document.getElementById('latency-eff');
    
    // Xception display elements
    const scoreXcp = document.getElementById('score-xcp');
    const labelXcp = document.getElementById('label-xcp');
    const circleXcp = document.getElementById('circle-xcp');
    const heatmapXcp = document.getElementById('heatmap-xcp');
    const latencyXcp = document.getElementById('latency-xcp');
    
    // Hybrid display elements
    const scoreHyb = document.getElementById('score-hyb');
    const labelHyb = document.getElementById('label-hyb');
    const circleHyb = document.getElementById('circle-hyb');
    const latencyHyb = document.getElementById('latency-hyb');
    const hybridActiveDisplay = document.getElementById('hybrid-active-display');
    const hybridInactiveDisplay = document.getElementById('hybrid-inactive-display');
    const hybridLatencyPill = document.getElementById('hybrid-latency-pill');
    const hybridDescBox = document.getElementById('hybrid-desc-box');

    // --- DRAG & DROP PIPELINE ---
    
    // Prevent defaults on drag/drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileDropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Add visual cues
    ['dragenter', 'dragover'].forEach(eventName => {
        fileDropzone.addEventListener(eventName, () => {
            fileDropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileDropzone.addEventListener(eventName, () => {
            fileDropzone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    fileDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Handle click file browse input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        selectedFileName.textContent = file.name;
        
        // Format size
        const sizeKB = file.size / 1024;
        if (sizeKB > 1024) {
            selectedFileSize.textContent = `${(sizeKB / 1024).toFixed(1)} MB`;
        } else {
            selectedFileSize.textContent = `${sizeKB.toFixed(1)} KB`;
        }
        
        // Show correct icon based on type
        if (file.type.startsWith('video/')) {
            filePreviewIcon.className = 'fa-regular fa-file-video file-type-icon';
            filePreviewIcon.style.color = 'var(--accent-purple)';
        } else if (file.type.startsWith('image/')) {
            filePreviewIcon.className = 'fa-regular fa-file-image file-type-icon';
            filePreviewIcon.style.color = 'var(--accent-cyan)';
        } else {
            filePreviewIcon.className = 'fa-regular fa-file file-type-icon';
            filePreviewIcon.style.color = 'var(--text-muted)';
        }
        
        // Toggle view
        dropzoneDefault.style.display = 'none';
        dropzoneActive.style.display = 'flex';
    }

    // Reset Playground
    btnReset.addEventListener('click', resetPlayground);
    
    function resetPlayground() {
        selectedFile = null;
        fileInput.value = '';
        videoUrlInput.value = '';
        dropzoneActive.style.display = 'none';
        dropzoneDefault.style.display = 'flex';
        analysisStatus.style.display = 'none';
        emptyStateResults.style.display = 'block';
        resultsContent.style.display = 'none';
        
        // Reset empty state text in case it showed an error
        const h3 = emptyStateResults.querySelector('h3');
        const p = emptyStateResults.querySelector('p');
        const icon = emptyStateResults.querySelector('i');
        if (icon) icon.className = 'fa-solid fa-magnifying-glass-chart';
        if (h3) h3.textContent = "Awaiting Inference";
        if (p) p.textContent = "Upload a file and trigger the analysis to see the models' predictions, latencies, and spatial attention heatmaps side-by-side.";
        
        // Reset status steps
        resetStatusSteps();
    }

    function resetStatusSteps() {
        progressBarFill.style.width = '0%';
        stepUpload.className = 'step-active';
        stepUpload.querySelector('i').className = 'fa-solid fa-circle-notch fa-spin';
        
        [stepFace, stepSpatial, stepTemporal].forEach(step => {
            step.className = 'step-pending';
            step.querySelector('i').className = 'fa-solid fa-circle-notch';
        });
    }

    // --- TAB SWITCHING ---
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabUrl.classList.remove('active');
        uploadContainer.style.display = 'block';
        urlContainer.style.display = 'none';
        resetPlayground();
    });

    tabUrl.addEventListener('click', () => {
        tabUrl.classList.add('active');
        tabUpload.classList.remove('active');
        urlContainer.style.display = 'block';
        uploadContainer.style.display = 'none';
        resetPlayground();
    });

    // --- SCAN VIDEO URL ---
    btnScanUrl.addEventListener('click', async () => {
        const url = videoUrlInput.value.trim();
        if (!url) {
            alert('Please enter a valid video URL.');
            return;
        }

        // 1. Setup status displays
        analysisStatus.style.display = 'block';
        emptyStateResults.style.display = 'block';
        resultsContent.style.display = 'none';
        resetStatusSteps();

        try {
            // Step 1: Connecting and validating URL
            updateStepState(stepUpload, 'completed');
            updateStepState(stepFace, 'active');
            progressBarFill.style.width = '25%';

            // Send API Request
            const response = await fetch('/api/analyze-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: url })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'API execution failed.');
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.message || 'Detection failed');
            }

            // Step 2: Face detected
            updateStepState(stepFace, 'completed');
            progressBarFill.style.width = '50%';

            if (!result.face_detected) {
                showError("No face detected in the video URL. Please ensure a face is clearly visible.");
                return;
            }

            // Step 3: Spatial complete
            updateStepState(stepSpatial, 'completed');
            progressBarFill.style.width = '75%';

            // Step 4: Temporal sequence process
            updateStepState(stepTemporal, 'completed');
            progressBarFill.style.width = '100%';

            setTimeout(() => {
                displayResults(result);
            }, 300);

        } catch (error) {
            console.error("URL Analysis Error:", error);
            showError(error.message);
        }
    });


    // --- TRIGGER INFERENCE PIPELINE ---
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        // 1. Setup status displays
        analysisStatus.style.display = 'block';
        emptyStateResults.style.display = 'block';
        resultsContent.style.display = 'none';
        resetStatusSteps();
        
        // Prepare multipart data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        try {
            // Step 1: Uploading complete
            updateStepState(stepUpload, 'completed');
            updateStepState(stepFace, 'active');
            progressBarFill.style.width = '25%';
            
            // Send API Request
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'API execution failed.');
            }
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.message || 'Detection failed');
            }
            
            // Step 2: Face detected
            updateStepState(stepFace, 'completed');
            progressBarFill.style.width = '50%';
            
            if (!result.face_detected) {
                showError("No face detected in the upload. Please verify that a clear face is visible in the video or image.");
                return;
            }
            
            // Step 3: Spatial complete
            updateStepState(stepSpatial, 'completed');
            progressBarFill.style.width = '75%';
            
            // Step 4: Temporal sequence process
            updateStepState(stepTemporal, 'completed');
            progressBarFill.style.width = '100%';
            
            setTimeout(() => {
                displayResults(result);
            }, 300);
            
        } catch (error) {
            console.error("Analysis Error:", error);
            showError(error.message);
        }
    });

    function updateStepState(element, state) {
        if (state === 'active') {
            element.className = 'step-active';
            element.querySelector('i').className = 'fa-solid fa-circle-notch fa-spin';
        } else if (state === 'completed') {
            element.className = 'step-completed';
            element.querySelector('i').className = 'fa-solid fa-circle-check';
        }
    }

    function showError(message) {
        emptyStateResults.style.display = 'block';
        resultsContent.style.display = 'none';
        
        const h3 = emptyStateResults.querySelector('h3');
        const p = emptyStateResults.querySelector('p');
        const icon = emptyStateResults.querySelector('i');
        
        icon.className = 'fa-solid fa-circle-exclamation text-danger';
        h3.textContent = "Pipeline Error";
        p.textContent = message;
    }

    function displayResults(data) {
        // Toggle layouts
        emptyStateResults.style.display = 'none';
        resultsContent.style.display = 'flex';
        
        // 1. Cropped face anchor
        croppedFaceImg.src = data.cropped_face;
        
        // 2. Set spatial model results
        const eff = data.predictions.efficientnet;
        const xcp = data.predictions.xception;
        const hyb = data.predictions.hybrid;
        
        // Render EfficientNet B0 Card
        scoreEff.textContent = `${eff.confidence.toFixed(1)}%`;
        labelEff.textContent = eff.class;
        labelEff.className = `verdict-label ${eff.class.toLowerCase()}`;
        heatmapEff.src = eff.heatmap;
        latencyEff.textContent = data.latency_ms.efficientnet;
        updateCircleProgress(circleEff, eff.confidence, eff.class);
        
        // Render Xception Card
        scoreXcp.textContent = `${xcp.confidence.toFixed(1)}%`;
        labelXcp.textContent = xcp.class;
        labelXcp.className = `verdict-label ${xcp.class.toLowerCase()}`;
        heatmapXcp.src = xcp.heatmap;
        latencyXcp.textContent = data.latency_ms.xception;
        updateCircleProgress(circleXcp, xcp.confidence, xcp.class);
        
        // Render Hybrid Card
        if (data.file_type === 'video' && hyb) {
            hybridActiveDisplay.style.display = 'flex';
            hybridInactiveDisplay.style.display = 'none';
            hybridLatencyPill.style.display = 'flex';
            hybridDescBox.style.display = 'block';
            
            scoreHyb.textContent = `${hyb.confidence.toFixed(1)}%`;
            labelHyb.textContent = hyb.class;
            labelHyb.className = `verdict-label ${hyb.class.toLowerCase()}`;
            latencyHyb.textContent = data.latency_ms.hybrid;
            updateCircleProgress(circleHyb, hyb.confidence, hyb.class);
        } else {
            // Disabled state
            hybridActiveDisplay.style.display = 'none';
            hybridInactiveDisplay.style.display = 'flex';
            hybridLatencyPill.style.display = 'none';
            hybridDescBox.style.display = 'none';
        }
        
        // 3. Compute Consensus Verdict
        let scoreSum = 0;
        let modelCount = 0;
        let fakeCount = 0;
        
        // Xception
        scoreSum += (xcp.class === 'FAKE') ? xcp.confidence : (100 - xcp.confidence);
        modelCount++;
        if (xcp.class === 'FAKE') fakeCount++;
        
        // EfficientNet
        scoreSum += (eff.class === 'FAKE') ? eff.confidence : (100 - eff.confidence);
        modelCount++;
        if (eff.class === 'FAKE') fakeCount++;
        
        // Hybrid
        if (data.file_type === 'video' && hyb) {
            scoreSum += (hyb.class === 'FAKE') ? hyb.confidence : (100 - hyb.confidence);
            modelCount++;
            if (hyb.class === 'FAKE') fakeCount++;
        }
        
        const averageFakeProbability = scoreSum / modelCount;
        const consensusFake = averageFakeProbability >= 50;
        const consensusConfidence = consensusFake ? averageFakeProbability : (100 - averageFakeProbability);
        
        consensusBadge.textContent = consensusFake ? "FAKE" : "REAL";
        consensusBadge.className = `verdict-badge ${consensusFake ? 'fake' : 'real'}`;
        consensusDetails.textContent = `Consensus classification confidence: ${consensusConfidence.toFixed(1)}%`;
    }

    function updateCircleProgress(circleEl, percent, category) {
        const radius = 40;
        const circumference = 2 * Math.PI * radius; // 251.2
        const offset = circumference - (percent / 100) * circumference;
        
        circleEl.style.strokeDashoffset = offset;
        circleEl.style.stroke = (category === 'FAKE') ? 'var(--accent-red)' : 'var(--accent-cyan)';
    }

    // --- ACADEMIC METADATA FETCH & CHART.JS ---
    async function loadAcademicMetrics() {
        try {
            const response = await fetch('/api/model-metadata');
            if (!response.ok) throw new Error("Metadata request failed.");
            
            const metrics = await response.json();
            
            // 1. Render Summary Table
            renderSummaryTable(metrics);
            
            // 2. Render Charts
            renderCharts(metrics);
            
        } catch (error) {
            console.error("Metadata Loading Error:", error);
            const tbody = document.getElementById('metrics-table-body');
            tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger"><i class="fa-solid fa-triangle-exclamation"></i> Error loading metrics from server.</td></tr>`;
        }
    }

    function renderSummaryTable(data) {
        const tbody = document.getElementById('metrics-table-body');
        tbody.innerHTML = '';
        
        Object.keys(data).forEach(key => {
            const m = data[key];
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${m.name}</strong></td>
                <td><span class="text-success" style="font-weight: 700">${m.accuracy}%</span></td>
                <td>${(m.loss !== null && m.loss !== undefined) ? m.loss.toFixed(4) : '<span class="text-muted">N/A</span>'}</td>
                <td>${(m.auc !== null && m.auc !== undefined) ? m.auc.toFixed(3) : '<span class="text-muted">N/A</span>'}</td>
                <td>${m.parameters_m}M</td>
                <td>${m.latency_ms}ms</td>
            `;
            tbody.appendChild(row);
        });
    }

    function renderCharts(data) {
        const models = Object.keys(data).map(key => data[key].name);
        const accuracies = Object.keys(data).map(key => data[key].accuracy);
        const parameters = Object.keys(data).map(key => data[key].parameters_m);
        const latencies = Object.keys(data).map(key => data[key].latency_ms);
        
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'hsl(218, 15%, 75%)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'hsl(218, 15%, 75%)' }
                }
            }
        };

        // Chart 1: Accuracies
        const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
        accuracyChart = new Chart(ctxAcc, {
            type: 'bar',
            data: {
                labels: ['EfficientNetB0', 'Xception', 'Hybrid Model'],
                datasets: [{
                    data: accuracies,
                    backgroundColor: [
                        'rgba(6, 182, 212, 0.65)',
                        'rgba(168, 85, 247, 0.65)',
                        'rgba(244, 63, 94, 0.65)'
                    ],
                    borderColor: [
                        '#06b6d4',
                        '#a855f7',
                        '#f43f5e'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 8
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        min: 90,
                        max: 100,
                        ticks: {
                            ...chartOptions.scales.y.ticks,
                            callback: value => `${value}%`
                        }
                    }
                }
            }
        });

        // Chart 2: Parameters
        const ctxParams = document.getElementById('parametersChart').getContext('2d');
        parametersChart = new Chart(ctxParams, {
            type: 'bar',
            data: {
                labels: ['EfficientNetB0', 'Xception', 'Hybrid Model'],
                datasets: [{
                    data: parameters,
                    backgroundColor: [
                        'rgba(6, 182, 212, 0.65)',
                        'rgba(168, 85, 247, 0.65)',
                        'rgba(244, 63, 94, 0.65)'
                    ],
                    borderColor: [
                        '#06b6d4',
                        '#a855f7',
                        '#f43f5e'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 8
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        ticks: {
                            ...chartOptions.scales.y.ticks,
                            callback: value => `${value}M`
                        }
                    }
                }
            }
        });

        // Chart 3: Latency
        const ctxLat = document.getElementById('latencyChart').getContext('2d');
        latencyChart = new Chart(ctxLat, {
            type: 'bar',
            data: {
                labels: ['EfficientNetB0', 'Xception', 'Hybrid Model'],
                datasets: [{
                    data: latencies,
                    backgroundColor: [
                        'rgba(6, 182, 212, 0.65)',
                        'rgba(168, 85, 247, 0.65)',
                        'rgba(244, 63, 94, 0.65)'
                    ],
                    borderColor: [
                        '#06b6d4',
                        '#a855f7',
                        '#f43f5e'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 8
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        ticks: {
                            ...chartOptions.scales.y.ticks,
                            callback: value => `${value}ms`
                        }
                    }
                }
            }
        });
    }

    // Trigger metadata fetch on initial startup load
    loadAcademicMetrics();
});
