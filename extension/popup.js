document.addEventListener('DOMContentLoaded', () => {
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');
    const serverError = document.getElementById('server-error');
    const btnScanPage = document.getElementById('btn-scan-page');
    const extensionUrlInput = document.getElementById('extension-url-input');
    const btnScanUrlExt = document.getElementById('btn-scan-url-ext');
    
    const resultPane = document.getElementById('result-pane');
    const verdictCard = document.getElementById('verdict-card');
    const verdictTag = document.getElementById('verdict-tag');
    const verdictScore = document.getElementById('verdict-score');
    const verdictDesc = document.getElementById('verdict-desc');

    const BACKEND_URL = 'http://127.0.0.1:8000';

    // --- CHECK BACKEND CONNECTION ---
    async function checkServerConnection() {
        try {
            const response = await fetch(`${BACKEND_URL}/api/model-metadata`, {
                method: 'GET',
                // Avoid caching issues
                cache: 'no-store'
            });
            
            if (response.ok) {
                // Connection successful
                connectionDot.className = 'status-dot connected';
                connectionText.textContent = 'Server: Connected';
                serverError.style.display = 'none';
                btnScanPage.disabled = false;
                extensionUrlInput.disabled = false;
                btnScanUrlExt.disabled = false;
            } else {
                throw new Error("Metadata endpoint returned error");
            }
        } catch (error) {
            console.warn("Connection check failed:", error);
            connectionDot.className = 'status-dot disconnected';
            connectionText.textContent = 'Server: Offline';
            serverError.style.display = 'flex';
            btnScanPage.disabled = true;
            extensionUrlInput.disabled = true;
            btnScanUrlExt.disabled = true;
            resultPane.style.display = 'none';
        }
    }

    // Ping server immediately on open
    checkServerConnection();
    // Re-check every 3 seconds while popup is open
    const connectionTimer = setInterval(checkServerConnection, 3000);

    // Stop timer when window closes
    window.addEventListener('blur', () => {
        clearInterval(connectionTimer);
    });

    // --- SCAN ACTIVE TAB VIDEO ---
    btnScanPage.addEventListener('click', async () => {
        btnScanPage.disabled = true;
        btnScanPage.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Triggering Scan...';
        
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (!tab) {
                throw new Error("No active browser tab found.");
            }

            // Send message to content script in the active tab to find and scan the video
            chrome.tabs.sendMessage(tab.id, { action: "scan_active_video" }, (response) => {
                // Handle callback responses from the content script
                btnScanPage.disabled = false;
                btnScanPage.innerHTML = '<i class="fa-solid fa-expand"></i> Scan Active Video';
                
                if (chrome.runtime.lastError) {
                    alert("Unable to communicate with the page. Reload the tab and ensure a video is visible.");
                    return;
                }

                if (!response) {
                    alert("No response received from the page.");
                    return;
                }

                if (!response.success) {
                    alert(response.message || "Failed to scan page video.");
                    return;
                }

                // Render result
                displayScanResult(response.result);
            });

        } catch (err) {
            console.error(err);
            btnScanPage.disabled = false;
            btnScanPage.innerHTML = '<i class="fa-solid fa-expand"></i> Scan Active Video';
            alert(err.message || "Error starting page scan.");
        }
    });

    // --- SCAN DIRECT VIDEO URL ---
    async function scanDirectVideoUrl() {
        const url = extensionUrlInput.value.trim();
        if (!url) {
            alert("Please enter a video URL.");
            return;
        }

        btnScanUrlExt.disabled = true;
        extensionUrlInput.disabled = true;
        btnScanUrlExt.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

        try {
            const response = await fetch(`${BACKEND_URL}/api/analyze-url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: url })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "URL scan failed.");
            }

            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.message || "Failed to scan URL.");
            }

            // Map standard comparative result to extension layout
            const mappedResult = {
                face_detected: data.face_detected,
                class: data.predictions.hybrid ? data.predictions.hybrid.class : data.predictions.xception.class,
                confidence: data.predictions.hybrid ? data.predictions.hybrid.confidence : data.predictions.xception.confidence
            };

            displayScanResult(mappedResult);

        } catch (err) {
            console.error(err);
            alert(err.message || "Error analyzing video URL.");
        } finally {
            btnScanUrlExt.disabled = false;
            extensionUrlInput.disabled = false;
            btnScanUrlExt.innerHTML = '<i class="fa-solid fa-arrow-right"></i>';
        }
    }

    btnScanUrlExt.addEventListener('click', scanDirectVideoUrl);
    extensionUrlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            scanDirectVideoUrl();
        }
    });

    function displayScanResult(result) {
        resultPane.style.display = 'block';
        
        // Remove old classes
        verdictCard.className = 'verdict-card';
        
        if (!result.face_detected) {
            verdictCard.classList.add('fake'); // Style as error
            verdictTag.textContent = 'ERROR';
            verdictScore.textContent = '';
            verdictDesc.textContent = 'No face detected in video player.';
            return;
        }

        const isFake = result.class === 'FAKE';
        verdictCard.classList.add(isFake ? 'fake' : 'real');
        
        verdictTag.textContent = result.class;
        verdictScore.textContent = `${result.confidence.toFixed(1)}% Confidence`;
        
        if (isFake) {
            verdictDesc.textContent = 'WARNING: Sequential jitter & blending defects detected.';
        } else {
            verdictDesc.textContent = 'SECURE: Spatio-temporal movements are biologically consistent.';
        }
    }
});
