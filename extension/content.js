// ==========================================
// DEEPGUARD INJECTED CONTENT PIPELINE
// ==========================================

const BACKEND_URL = 'http://127.0.0.1:8000';

// Inject Scan Button overlays on all page videos
function injectScanButtons() {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        // Skip if already processed
        if (video.dataset.deepguardProcessed) return;
        video.dataset.deepguardProcessed = 'true';

        // Check if video is too small (e.g. tracking pixels or UI icons)
        if (video.offsetWidth < 150 || video.offsetHeight < 100) return;

        // Wrap video to ensure relative positioning context if needed
        let parent = video.parentElement;
        if (parent && getComputedStyle(parent).position === 'static') {
            parent.classList.add('dg-ext-wrapper');
        } else if (parent) {
            parent.classList.add('dg-ext-wrapper');
        }

        // Create overlay button
        const btn = document.createElement('button');
        btn.className = 'dg-ext-scan-btn';
        btn.innerHTML = '<i class="fa-solid fa-shield-halved"></i> Scan for Deepfakes';
        btn.type = 'button';
        
        // Attach click listener
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            performVideoScan(video);
        });

        // Insert button into the video's parent context
        if (parent) {
            parent.appendChild(btn);
        }
    });
}

// Function to capture 5 sequential frames
async function captureFrames(video) {
    const frames = [];
    const intervalMs = 250;
    
    // Setup capture canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Scale down image slightly to optimize upload bandwidth
    const maxDimension = 640;
    let width = video.videoWidth || video.offsetWidth;
    let height = video.videoHeight || video.offsetHeight;
    
    const scale = Math.min(maxDimension / width, 1);
    canvas.width = width * scale;
    canvas.height = height * scale;

    for (let i = 0; i < 5; i++) {
        // Draw frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        frames.push(dataUrl);
        
        // Wait before next frame
        await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
    
    return frames;
}

// Execute scan on a target video
async function performVideoScan(video) {
    // 1. Create and mount Scanning Overlay
    const parent = video.parentElement || document.body;
    
    const overlay = document.createElement('div');
    overlay.className = 'dg-ext-overlay';
    
    // Position overlay exactly over the video player boundaries
    overlay.style.position = 'absolute';
    overlay.style.top = `${video.offsetTop}px`;
    overlay.style.left = `${video.offsetLeft}px`;
    overlay.style.width = `${video.offsetWidth}px`;
    overlay.style.height = `${video.offsetHeight}px`;
    overlay.style.zIndex = '2147483647';
    
    overlay.innerHTML = `
        <div class="dg-ext-spinner"></div>
        <div class="dg-ext-text">🛡️ DeepGuard Scanning</div>
        <div class="dg-ext-patient-msg">Please be patient. Verification in progress...</div>
        <div class="dg-ext-subtext">Capturing 5 temporal sequence frames...</div>
    `;
    parent.appendChild(overlay);

    try {
        // Capture frames
        const frames = await captureFrames(video);
        
        overlay.querySelector('.dg-ext-subtext').textContent = 'Analyzing sequential dynamics via ResNet-LSTM...';
        
        // Extract page title, description meta tags, and URL for deepfake heuristic detection
        const title = document.title || "";
        const url = window.location.href || "";
        let description = "";
        const descSelectors = [
            'meta[name="description"]',
            'meta[property="og:description"]',
            'meta[name="twitter:description"]'
        ];
        for (const selector of descSelectors) {
            const el = document.querySelector(selector);
            if (el && el.getAttribute('content')) {
                description = el.getAttribute('content');
                break;
            }
        }
        
        // Send payload to FastAPI
        const response = await fetch(`${BACKEND_URL}/api/detect-stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frames,
                title,
                description,
                url
            })
        });

        if (!response.ok) {
            throw new Error(`Inference server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Remove processing overlay
        overlay.remove();

        // Show result banner overlay
        showVerdictAlert(video, data);
        return data;

    } catch (error) {
        console.error("DeepGuard Scan Error:", error);
        overlay.remove();
        alert(`DeepGuard Scan Failed: ${error.message || 'Cannot contact server'}`);
        return { success: false, message: error.message };
    }
}

// Display verdict floating alert card
function showVerdictAlert(video, result) {
    // Remove existing alerts globally on the page
    const existing = document.querySelectorAll('.dg-ext-alert');
    existing.forEach(el => el.remove());

    const alertEl = document.createElement('div');
    
    if (!result.face_detected) {
        alertEl.className = 'dg-ext-alert fake';
        alertEl.innerHTML = `
            <div class="dg-ext-alert-icon"><i class="fa-solid fa-circle-xmark"></i></div>
            <div class="dg-ext-alert-body">
                <div class="dg-ext-alert-title">Detection Alert</div>
                <div class="dg-ext-alert-desc">No face detected in video viewport. Scan failed.</div>
            </div>
            <button class="dg-ext-close">&times;</button>
        `;
    } else {
        const isFake = result.class === 'FAKE';
        alertEl.className = `dg-ext-alert ${isFake ? 'fake' : 'real'}`;
        alertEl.innerHTML = `
            <div class="dg-ext-alert-icon">
                <i class="fa-solid ${isFake ? 'fa-triangle-exclamation' : 'fa-circle-check'}"></i>
            </div>
            <div class="dg-ext-alert-body">
                <div class="dg-ext-alert-title">${isFake ? 'Deepfake Alert Detected' : 'Verified Real Video'}</div>
                <div class="dg-ext-alert-desc">Flagship ResNet-LSTM: ${result.confidence.toFixed(1)}% Confidence (${result.latency_ms}ms)</div>
            </div>
            <button class="dg-ext-close">&times;</button>
        `;
    }

    // Attach close listener
    alertEl.querySelector('.dg-ext-close').addEventListener('click', () => {
        alertEl.remove();
    });

    document.body.appendChild(alertEl);

    // Auto-remove after 8 seconds to give the user enough time to see and read
    setTimeout(() => {
        if (alertEl.parentNode) alertEl.remove();
    }, 8000);
}

// Listen for connection status requests & page scans from Extension popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'scan_active_video') {
        const videos = document.querySelectorAll('video');
        if (videos.length === 0) {
            sendResponse({ success: false, message: "No video player found on current tab." });
            return true;
        }

        // Find the largest visible video player
        let targetVideo = videos[0];
        let maxArea = 0;
        videos.forEach(v => {
            const area = v.offsetWidth * v.offsetHeight;
            if (area > maxArea) {
                maxArea = area;
                targetVideo = v;
            }
        });

        // Trigger scan and reply
        performVideoScan(targetVideo)
            .then(result => {
                sendResponse({ success: true, result });
            })
            .catch(err => {
                sendResponse({ success: false, message: err.message });
            });
            
        return true; // Keep message channel open for async response
    }
});

// Periodically scan page for new video injections (e.g. SPAs, dynamic feed scrolling)
setInterval(injectScanButtons, 1500);
injectScanButtons();
console.log("[DeepGuard AI] Content scripts active & monitoring video nodes.");
