// MediaPipe Hands and Camera utility will be loaded dynamically from CDN

import * as THREE from 'three';
import * as zarr from 'zarrita';

// Zarr file path
const ZARR_PATH = 'https://raw.githubusercontent.com/JEFworks-Lab/CellCarto-OvulationSlideseq/refs/heads/main/adata_ovary_combined_fullprocessed_annotated_withglobalspatial.zarr';

// Global variables
let scene, camera, renderer, controls;
let points, pointsGeometry, pointsMaterial;
let pointsGroup; // Group to hold spheres for rotation
let instancedMesh; // InstancedMesh for spheres
let embeddings = {
    global_spatial: null,
    umap: null,
    pca: null,
    celltypes: null,
    celltype_colors: null
};
let currentMode = 'global_spatial';
let handDetectionActive = false;
let hands;
let video;
let videoCanvas;
let videoCanvasCtx;
let detectedGesture = 'None';
let detectionDebug = '';
let palmRotation = 0; // Stores PC start index for PCA visualization (set by finger count, NOT hand rotation)
let currentRotationAngle = 0; // Current camera rotation angle in degrees (for UI display)

// Animation state for smooth transitions
let isAnimating = false;
let sourcePositions = null;
let targetPositions = null;
let animationStartTime = 0;
let animationDuration = 1000; // 1 second animation
let cameraRotationY = 0; // Current rotation around z-axis (in x-y plane)
let targetRotationY = 0; // Target rotation (smoothly interpolated to)
let cameraDistance = 2; // Current camera distance
let targetCameraDistance = 2; // Target camera distance (smoothly interpolated to)
const SPHERE_RADIUS = 0.005; // Radius of each sphere (increased for visibility)
const ROTATION_SMOOTHING = 0.01; // Smoothing factor for rotation (0-1, lower = smoother but slower)
const ZOOM_SMOOTHING = 0.01; // Smoothing factor for zoom (0-1, lower = smoother but slower)
const MIN_PALM_DISTANCE = 0.03; // Minimum palm distance (hand far from camera) - maps to max zoom out
const MAX_PALM_DISTANCE = 0.12; // Maximum palm distance (hand close to camera) - maps to max zoom in
const MIN_CAMERA_DISTANCE = 0.5; // Minimum camera distance (zoomed in)
const MAX_CAMERA_DISTANCE = 10; // Maximum camera distance (zoomed out)

// Finger count detection
let fingerCount = 0; // 0-5 fingers
let currentHandDistance = 0; // Current detected hand distance (for display)
let currentHandAngleDeg = 0; // Current detected hand angle in degrees (for display)
let totalCellsLoaded = 0; // Total number of cells loaded (for display)

// Initialize Three.js scene
function initScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    const canvas = document.getElementById('canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    
    // Helper function to get right column dimensions
    const getRightColumnSize = () => {
        const rightColumn = document.getElementById('right-column');
        if (rightColumn) {
            return {
                width: rightColumn.clientWidth,
                height: rightColumn.clientHeight
            };
        }
        return { width: window.innerWidth, height: window.innerHeight };
    };
    
    // Set initial renderer size
    const { width: initialWidth, height: initialHeight } = getRightColumnSize();
    renderer.setSize(initialWidth, initialHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Camera setup - centered above the x-y plane
    camera = new THREE.PerspectiveCamera(
        75,
        initialWidth / initialHeight,
        0.01,
        10000
    );
    // Position camera centered above the x-y plane looking down
    camera.position.set(0, 0, cameraDistance);
    camera.lookAt(0, 0, 0);
    
    // Create a group for the points that can be rotated
    pointsGroup = new THREE.Group();
    scene.add(pointsGroup);
    
    window.addEventListener('resize', onWindowResize);
}

function updateCameraPosition() {
    // Camera is centered above the x-y plane, looking straight down
    camera.position.set(0, 0, cameraDistance);
    camera.lookAt(0, 0, 0);
    
    // Rotate the points group around z-axis based on hand rotation
    if (pointsGroup) {
        pointsGroup.rotation.z = -cameraRotationY; // Negative to match expected rotation direction
    }
}

function onWindowResize() {
    // Get right column dimensions for canvas sizing
    const rightColumn = document.getElementById('right-column');
    if (rightColumn) {
        const width = rightColumn.clientWidth;
        const height = rightColumn.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    } else {
        // Fallback to window size if right column not found
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Helper function to load a zarr array
async function loadZarrArray(root, path) {
    try {
        const arr = await zarr.open(root.resolve(path), { kind: 'array' });
        const data = await zarr.get(arr);
        return data;
    } catch (error) {
        console.error(`[loadZarrArray] Error loading ${path}:`, error);
        throw error;
    }
}

// Helper function to load categorical column (codes + categories)
async function loadCategoricalColumn(root, basePath) {
    const codesArr = await zarr.open(root.resolve(`${basePath}/codes`), { kind: 'array' });
    const catsArr = await zarr.open(root.resolve(`${basePath}/categories`), { kind: 'array' });
    
    const codes = await zarr.get(codesArr);
    const categories = await zarr.get(catsArr);
    
    // Map codes to category values
    const result = new Array(codes.data.length);
    for (let i = 0; i < codes.data.length; i++) {
        result[i] = categories.data[codes.data[i]];
    }
    return result;
}

// Normalize array to [-1, 1] range
function normalize(arr) {
    const n = arr.length;
    if (n === 0) return arr;
    
    const dims = arr[0].length;
    const normalized = [];
    
    // Calculate min/max for each dimension
    const mins = new Array(dims).fill(Infinity);
    const maxs = new Array(dims).fill(-Infinity);
    
    for (let i = 0; i < n; i++) {
        for (let d = 0; d < dims; d++) {
            mins[d] = Math.min(mins[d], arr[i][d]);
            maxs[d] = Math.max(maxs[d], arr[i][d]);
        }
    }
    
    // Normalize
    for (let i = 0; i < n; i++) {
        const normalizedPoint = [];
        for (let d = 0; d < dims; d++) {
            const range = maxs[d] - mins[d];
            if (range === 0) {
                normalizedPoint[d] = 0;
            } else {
                normalizedPoint[d] = 2 * (arr[i][d] - mins[d]) / range - 1;
            }
        }
        normalized.push(normalizedPoint);
    }
    
    return normalized;
}

// Generate colors for celltypes (rainbow palette)
function generateCelltypeColors(celltypes) {
    if (!celltypes) return {};
    
    const unique_celltypes = [...new Set(celltypes)].sort();
    const n_unique = unique_celltypes.length;
    const celltype_colors = {};
    
    for (let i = 0; i < n_unique; i++) {
        const celltype = unique_celltypes[i];
        const hue = i / n_unique; // Evenly distribute from 0 to 1
        const saturation = 0.8;
        const value = 0.9;
        
        // HSV to RGB conversion
        const c = value * saturation;
        const x = c * (1 - Math.abs((hue * 6) % 2 - 1));
        const m = value - c;
        
        let r, g, b;
        if (hue < 1/6) {
            r = c; g = x; b = 0;
        } else if (hue < 2/6) {
            r = x; g = c; b = 0;
        } else if (hue < 3/6) {
            r = 0; g = c; b = x;
        } else if (hue < 4/6) {
            r = 0; g = x; b = c;
        } else if (hue < 5/6) {
            r = x; g = 0; b = c;
        } else {
            r = c; g = 0; b = x;
        }
        
        const rgb = [r + m, g + m, b + m];
        const hex_color = `#${Math.round(rgb[0]*255).toString(16).padStart(2, '0')}${Math.round(rgb[1]*255).toString(16).padStart(2, '0')}${Math.round(rgb[2]*255).toString(16).padStart(2, '0')}`;
        
        celltype_colors[celltype] = {
            'hex': hex_color,
            'rgb': rgb
        };
    }
    
    return celltype_colors;
}

// Load embeddings directly from zarr file
async function loadEmbeddings() {
    try {
        updateStatus('Loading embeddings...');
        
        // Open zarr store
        const baseUrl = new URL(ZARR_PATH, window.location.href).href;
        const store = new zarr.FetchStore(baseUrl);
        const root = zarr.root(store);
        
        // Load Sample annotation first to create filter
        updateStatus('Loading Sample annotations...');
        let sampleFilter = null;
        try {
            const samples = await loadCategoricalColumn(root, 'obs/Sample');
            // Create boolean mask for "Untreated" sample
            sampleFilter = samples.map(s => s === 'Untreated');
            const untreatedCount = sampleFilter.filter(f => f).length;
            console.log(`Found ${untreatedCount} cells from "Untreated" sample out of ${samples.length} total cells`);
        } catch (e) {
            console.warn(`Error loading Sample: ${e}`);
            // If Sample column doesn't exist, don't filter
            sampleFilter = null;
        }
        
        // Load global spatial embedding (2D - x, y)
        updateStatus('Loading Global Spatial...');
        const global_x_data = await loadZarrArray(root, 'obsm/Global_Spatial/global_x');
        const global_y_data = await loadZarrArray(root, 'obsm/Global_Spatial/global_y');
        const global_x = Array.from(global_x_data.data);
        const global_y = Array.from(global_y_data.data);
        const n_cells = global_x.length;
        
        // Create 3D by using 0 for z
        let global_spatial = [];
        for (let i = 0; i < n_cells; i++) {
            global_spatial.push([global_x[i], global_y[i], 0]);
        }
        
        // Helper function to filter array based on sample filter
        function filterBySample(arr) {
            if (!sampleFilter || arr.length === 0) return arr;
            return arr.filter((_, i) => sampleFilter[i]);
        }
        
        // Filter global_spatial by sample
        global_spatial = filterBySample(global_spatial);
        
        // Load UMAP embedding from obsm/X_umap
        updateStatus('Loading UMAP...');
        let umap = null;
        try {
            const umap_array = await zarr.open(root.resolve('obsm/X_umap'), { kind: 'array' });
            const umap_data = await zarr.get(umap_array);
            
            // umap_data.data is a flat array, umap_data.shape is [n_cells, n_dims]
            // Data is stored in row-major order
            const n_dims = umap_data.shape[1] || 2;
            const umap_2d = [];
            for (let i = 0; i < n_cells; i++) {
                const idx = i * n_dims;
                umap_2d.push([
                    umap_data.data[idx] || 0,
                    umap_data.data[idx + 1] || 0,
                    0
                ]);
            }
            umap = umap_2d;
            console.log(`Loaded UMAP: shape=${umap.length}x${umap[0].length}`);
        } catch (e) {
            console.warn(`Error loading UMAP: ${e}`);
            umap = new Array(n_cells).fill(null).map(() => [0, 0, 0]);
        }
        
        // Filter UMAP by sample
        umap = filterBySample(umap);
        
        // Load PCA Harmony - first 5 components
        updateStatus('Loading PCA Harmony...');
        let pca = null;
        try {
            const pca_array = await zarr.open(root.resolve('obsm/X_pca_harmony'), { kind: 'array' });
            const pca_data = await zarr.get(pca_array);
            
            // pca_data.data is a flat array, pca_data.shape is [n_cells, n_dims]
            // Data is stored in row-major order
            const total_dims = pca_data.shape[1] || 5;
            const n_dims = Math.min(5, total_dims);
            const pca_5d = [];
            for (let i = 0; i < n_cells; i++) {
                const idx = i * total_dims;
                const point = [];
                for (let d = 0; d < n_dims; d++) {
                    point.push(pca_data.data[idx + d] || 0);
                }
                pca_5d.push(point);
            }
            pca = pca_5d;
            console.log(`Loaded PCA Harmony: shape=${pca.length}x${pca[0].length}`);
        } catch (e) {
            console.warn(`Error loading PCA Harmony: ${e}`);
            pca = new Array(n_cells).fill(null).map(() => new Array(5).fill(0));
        }
        
        // Filter PCA by sample
        pca = filterBySample(pca);
        
        // Load Celltypes annotation (categorical variable)
        updateStatus('Loading Celltypes...');
        let celltypes = null;
        try {
            celltypes = await loadCategoricalColumn(root, 'obs/Celltypes');
            console.log(`Loaded ${celltypes.length} cell annotations`);
            console.log(`First 10 celltypes:`, celltypes.slice(0, 10));
        } catch (e) {
            console.warn(`Error loading Celltypes: ${e}`);
            celltypes = null;
        }
        
        // Filter celltypes by sample
        if (celltypes) {
            celltypes = filterBySample(celltypes);
        }
        
        // Generate colors for celltypes
        const celltype_colors = generateCelltypeColors(celltypes);
        console.log(`Generated colors for ${Object.keys(celltype_colors).length} unique celltypes`);
        
        // Normalize all embeddings
        updateStatus('Normalizing embeddings...');
        embeddings.global_spatial = normalize(global_spatial);
        embeddings.umap = normalize(umap);
        embeddings.pca = normalize(pca);
        embeddings.celltypes = celltypes;
        embeddings.celltype_colors = celltype_colors;
        
        console.log(`Loaded ${embeddings.celltypes ? embeddings.celltypes.length : 0} cell annotations`);
        console.log(`Found ${embeddings.celltype_colors ? Object.keys(embeddings.celltype_colors).length : 0} unique celltypes`);
        
        // Update legend with cell types and colors
        updateLegend(celltype_colors);
        
        totalCellsLoaded = embeddings.global_spatial ? embeddings.global_spatial.length : 0;
        const uniqueCelltypes = embeddings.celltype_colors ? Object.keys(embeddings.celltype_colors).length : 0;
        
        // Create initial visualization with global spatial (no animation on first load)
        updateVisualization('global_spatial', null, null, false);
        updateStatus(`Ready: ${totalCellsLoaded.toLocaleString()} cells, ${uniqueCelltypes} cell types`);
    } catch (error) {
        console.error('Error loading embeddings:', error);
        updateStatus('Error loading data');
    }
}

// Update visualization based on mode (with animation)
function updateVisualization(mode, pcStart = null, pcEnd = null, animateTransition = true) {
    if (!embeddings[mode] && mode !== 'pca') return;
    if (mode === 'pca' && !embeddings.pca) return;
    
    // Get target positions
    let targetPos;
    if (mode === 'pca') {
        // Get the appropriate PC range based on finger count
        // pcStart and pcEnd are 0-indexed
        const startIdx = pcStart !== null ? pcStart : palmRotation;
        const endIdx = pcEnd !== null ? pcEnd : (startIdx + 1);
        
        targetPos = embeddings.pca.map(p => [
            p[startIdx] || 0,
            p[endIdx] || 0,
            0 // Pad with 0 for 3D
        ]);
        updateModeText(`PCs ${startIdx + 1}-${endIdx + 1}`);
    } else {
        // All embeddings are 2D, pad with 0 for z
        targetPos = embeddings[mode].map(p => [p[0] || 0, p[1] || 0, 0]);
        const modeNames = {
            'global_spatial': 'Global Spatial',
            'umap': 'UMAP'
        };
        updateModeText(modeNames[mode]);
    }
    
    // Get current positions for animation
    let currentPos = null;
    if (instancedMesh && instancedMesh.count > 0) {
        currentPos = [];
        const matrix = new THREE.Matrix4();
        for (let i = 0; i < instancedMesh.count; i++) {
            instancedMesh.getMatrixAt(i, matrix);
            const position = new THREE.Vector3();
            matrix.decompose(position, new THREE.Quaternion(), new THREE.Vector3());
            currentPos.push([position.x, position.y, position.z]);
        }
    }
    
    // If we have current positions and animation is enabled, start animation
    if (currentPos && currentPos.length === targetPos.length && animateTransition) {
        // Cancel any ongoing animation and start new one
        sourcePositions = currentPos;
        targetPositions = targetPos;
        animationStartTime = performance.now();
        isAnimating = true;
        currentMode = mode; // Update mode immediately
    } else {
        // No animation - set positions directly
        isAnimating = false; // Cancel any ongoing animation
        currentMode = mode;
        setPositions(targetPos);
    }
}

// Set positions directly (used when not animating or at end of animation)
function setPositions(positions) {
    // Remove old instanced mesh if it exists
    if (instancedMesh) {
        pointsGroup.remove(instancedMesh);
        instancedMesh.geometry.dispose();
        instancedMesh.material.dispose();
    }
    
    // Camera distance is controlled by hand gestures (targetCameraDistance),
    // not recalculated here, to ensure smooth transitions between embeddings
    
    // Create sphere geometry for instancing
    const sphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 8, 8);
    
    // Create instanced mesh
    instancedMesh = new THREE.InstancedMesh(
        sphereGeometry,
        null, // Material will be set below
        positions.length
    );
    
    // Color based on celltypes if available, otherwise use position-based colors
    const colors = new Float32Array(positions.length * 3);
    if (embeddings.celltypes && embeddings.celltype_colors && embeddings.celltypes.length === positions.length) {
        // Use celltype colors
        positions.forEach((pos, i) => {
            const celltype = embeddings.celltypes[i];
            const colorInfo = embeddings.celltype_colors[celltype];
            if (colorInfo && colorInfo.rgb) {
                // Use RGB values (0-1 range)
                colors[i * 3] = colorInfo.rgb[0];
                colors[i * 3 + 1] = colorInfo.rgb[1];
                colors[i * 3 + 2] = colorInfo.rgb[2];
            } else {
                // Fallback to gray if celltype not found
                colors[i * 3] = 0.5;
                colors[i * 3 + 1] = 0.5;
                colors[i * 3 + 2] = 0.5;
            }
        });
    } else {
        // Fallback: Color based on position for visual interest
        positions.forEach((pos, i) => {
            // Create gradient colors
            colors[i * 3] = (pos[0] + 1) / 2;
            colors[i * 3 + 1] = (pos[1] + 1) / 2;
            colors[i * 3 + 2] = (pos[2] + 1) / 2;
        });
    }
    // Set positions and colors for each instance
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    
    for (let i = 0; i < positions.length; i++) {
        const pos = positions[i];
        matrix.makeTranslation(pos[0], pos[1], pos[2]);
        instancedMesh.setMatrixAt(i, matrix);
        
        // Set color using setColorAt (proper way for InstancedMesh)
        const r = colors[i * 3];
        const g = colors[i * 3 + 1];
        const b = colors[i * 3 + 2];
        color.setRGB(r, g, b);
        instancedMesh.setColorAt(i, color);
    }
    
    // Create material for spheres with instanced colors
    const material = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.8
    });
    
    instancedMesh.material = material;
    instancedMesh.instanceMatrix.needsUpdate = true;
    if (instancedMesh.instanceColor) {
        instancedMesh.instanceColor.needsUpdate = true;
    }
    
    pointsGroup.add(instancedMesh);
    points = instancedMesh; // Keep reference for compatibility
}

// Update positions during animation
function updateAnimationPositions(positions) {
    if (!instancedMesh || !positions) return;
    
    const matrix = new THREE.Matrix4();
    for (let i = 0; i < positions.length && i < instancedMesh.count; i++) {
        const pos = positions[i];
        matrix.makeTranslation(pos[0], pos[1], pos[2]);
        instancedMesh.setMatrixAt(i, matrix);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
}

// Initialize MediaPipe Hands
async function initHandDetection() {
    video = document.getElementById('video');
    videoCanvas = document.getElementById('video-canvas');
    if (videoCanvas) {
        videoCanvasCtx = videoCanvas.getContext('2d');
        // Set initial canvas size
        videoCanvas.width = 640;
        videoCanvas.height = 480;
    }
    
    // Set canvas size to match video when video loads
    if (video) {
        video.addEventListener('loadedmetadata', () => {
            if (videoCanvas) {
                videoCanvas.width = video.videoWidth || 640;
                videoCanvas.height = video.videoHeight || 480;
            }
        });
    }
    
    // Load Hands from CDN dynamically if not already loaded
    if (typeof Hands === 'undefined') {
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
            script.onload = () => resolve();
            script.onerror = () => {
                // If CDN fails, try unpkg
                const script2 = document.createElement('script');
                script2.src = 'https://unpkg.com/@mediapipe/hands/hands.js';
                script2.onload = () => resolve();
                script2.onerror = reject;
                document.head.appendChild(script2);
            };
            document.head.appendChild(script);
        });
    }
    
    // Load Camera utility dynamically if not already loaded
    if (typeof Camera === 'undefined') {
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils';
            script.onload = () => resolve();
            script.onerror = () => {
                // If CDN fails, try unpkg
                const script2 = document.createElement('script');
                script2.src = 'https://unpkg.com/@mediapipe/camera_utils';
                script2.onload = () => resolve();
                script2.onerror = reject;
                document.head.appendChild(script2);
            };
            document.head.appendChild(script);
        });
    }
    
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });
    
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    hands.onResults(onHandResults);
    
    const camera = new Camera(video, {
        onFrame: async () => {
            await hands.send({ image: video });
        },
        width: 640,
        height: 480
    });
    
    camera.start();
    handDetectionActive = true;
    updateStatus('Camera ready - Show your hand!');
}

// Process hand detection results
function onHandResults(results) {
    // Clear canvas
    if (videoCanvasCtx) {
        videoCanvasCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
    }
    
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        // No hand detected - use global spatial
        detectedGesture = 'None';
        detectionDebug = 'No hand detected';
        currentHandDistance = 0;
        currentHandAngleDeg = 0;
        fingerCount = 0;
        if (currentMode !== 'global_spatial') {
            updateVisualization('global_spatial');
        }
        return;
    }
    
    const landmarks = results.multiHandLandmarks[0];
    
    // Draw hand landmarks and connections
    drawHandLandmarks(landmarks, results.multiHandedness?.[0]);
    
    // Hand rotation controls camera rotation ONLY (viewing angle in x-y plane, z-distance fixed)
    // This does NOT affect which embedding/PCs are used - that's controlled by finger count only
    const handRotation = detectHandRotation(landmarks);
    targetRotationY = handRotation; // Update target, will be smoothly interpolated in animate()
    
    // Hand distance controls camera zoom
    const handDistance = detectHandDistance(landmarks);
    targetCameraDistance = handDistance; // Update target, will be smoothly interpolated in animate()
    
    // Finger count controls which embedding/PCs to display (NOT hand rotation)
    const countResult = countFingers(landmarks);
    fingerCount = countResult.count;
    detectedGesture = `${fingerCount} finger${fingerCount !== 1 ? 's' : ''}`;
    
    // Update debug info (rotation angle for debugging)
    const rotationAngleDeg = (targetRotationY * 180 / Math.PI).toFixed(1);
    detectionDebug = `Fingers: ${fingerCount} (${countResult.fingers.join(', ')}) | Camera Rotation: ${rotationAngleDeg}°`;
    
    // Map finger count to visualization mode (PC selection based ONLY on finger count)
    if (fingerCount === 0) {
        // 0 fingers (fist) → Global Spatial
        if (currentMode !== 'global_spatial') {
            updateVisualization('global_spatial');
        }
    } else if (fingerCount === 1) {
        // 1 finger → UMAP
        if (currentMode !== 'umap') {
            updateVisualization('umap');
        }
    } else if (fingerCount >= 2 && fingerCount <= 5) {
        // 2-5 fingers → PCA (PC indices: fingerCount-1 to fingerCount)
        // 2 fingers → PC 1-2 (indices 0-1)
        // 3 fingers → PC 2-3 (indices 1-2)
        // 4 fingers → PC 3-4 (indices 2-3)
        // 5 fingers → PC 4-5 (indices 3-4)
        const pcStart = fingerCount - 2; // 0, 1, 2, or 3
        const pcEnd = fingerCount - 1;   // 1, 2, 3, or 4
        if (currentMode !== 'pca' || palmRotation !== pcStart) {
            palmRotation = pcStart; // Reuse this variable to store PC start index
            updateVisualization('pca', pcStart, pcEnd);
        }
    } else {
        // Unexpected finger count
        if (currentMode !== 'global_spatial') {
            updateVisualization('global_spatial');
        }
    }
}

// Draw hand landmarks and connections on video canvas
function drawHandLandmarks(landmarks, handedness) {
    if (!videoCanvasCtx || !landmarks) return;
    
    const ctx = videoCanvasCtx;
    const width = videoCanvas.width;
    const height = videoCanvas.height;
    
    // Draw connections (HAND_CONNECTIONS from MediaPipe)
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],        // Index
        [5, 9], [9, 10], [10, 11], [11, 12],   // Middle
        [9, 13], [13, 14], [14, 15], [15, 16], // Ring
        [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
    ];
    
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    
    connections.forEach(([start, end]) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];
        if (startPoint && endPoint) {
            ctx.beginPath();
            ctx.moveTo(startPoint.x * width, startPoint.y * height);
            ctx.lineTo(endPoint.x * width, endPoint.y * height);
            ctx.stroke();
        }
    });
    
    // Draw landmarks
    landmarks.forEach((landmark, index) => {
        const x = landmark.x * width;
        const y = landmark.y * height;
        
        // Draw different colors for fingertips and other points
        if ([4, 8, 12, 16, 20].includes(index)) {
            // Fingertips - draw larger and in red
            ctx.fillStyle = '#FF0000';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
        } else if (index === 0) {
            // Wrist - draw in yellow
            ctx.fillStyle = '#FFFF00';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
        } else {
            // Other points - draw smaller and in cyan
            ctx.fillStyle = '#00FFFF';
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
    
    // Draw handedness label
    if (handedness) {
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial';
        ctx.fillText(handedness.categoryName || 'Hand', 10, 20);
    }
}

// Count extended fingers
function countFingers(landmarks) {
    if (!landmarks || landmarks.length < 21) {
        return { count: 0, fingers: [] };
    }
    
    const fingers = [];
    let count = 0;
    
    // Thumb: Check if thumb tip (4) is to the right of thumb IP (3) when hand is facing camera
    // For right hand: thumb tip x > thumb IP x means extended
    // For left hand: thumb tip x < thumb IP x means extended
    // We'll use a more robust method: check if thumb tip is away from the hand
    const thumbTip = landmarks[4];
    const thumbIP = landmarks[3];
    const thumbMCP = landmarks[2];
    const thumbExtended = thumbTip.x > thumbIP.x; // Simplified: works for right hand
    if (thumbExtended) {
        fingers.push('Thumb');
        count++;
    }
    
    // Index finger: Check if index tip (8) is above index PIP (6)
    const indexTip = landmarks[8];
    const indexPIP = landmarks[6];
    const indexExtended = indexTip.y < indexPIP.y;
    if (indexExtended) {
        fingers.push('Index');
        count++;
    }
    
    // Middle finger: Check if middle tip (12) is above middle PIP (10)
    const middleTip = landmarks[12];
    const middlePIP = landmarks[10];
    const middleExtended = middleTip.y < middlePIP.y;
    if (middleExtended) {
        fingers.push('Middle');
        count++;
    }
    
    // Ring finger: Check if ring tip (16) is above ring PIP (14)
    const ringTip = landmarks[16];
    const ringPIP = landmarks[14];
    const ringExtended = ringTip.y < ringPIP.y;
    if (ringExtended) {
        fingers.push('Ring');
        count++;
    }
    
    // Pinky: Check if pinky tip (20) is above pinky PIP (18)
    const pinkyTip = landmarks[20];
    const pinkyPIP = landmarks[18];
    const pinkyExtended = pinkyTip.y < pinkyPIP.y;
    if (pinkyExtended) {
        fingers.push('Pinky');
        count++;
    }
    
    return { count, fingers };
}

// Detect hand distance from camera to control zoom
// Uses the distance between wrist and landmark[1] (thumb CMC) as a proxy for hand distance
// Larger distance = hand closer to camera = zoom in (lower camera distance)
function detectHandDistance(landmarks) {
    if (!landmarks || landmarks.length < 21) {
        return targetCameraDistance; // Return current distance if can't detect
    }
    
    const wrist = landmarks[0];
    const thumbCMC = landmarks[1]; // Thumb CMC joint (palm point)
    
    // Calculate 3D distance between wrist and palm point
    const palmDistance = distance3D(wrist, thumbCMC);
    
    // Store raw palm distance for display
    currentHandDistance = palmDistance;
    
    // Clamp palm distance to expected range
    const clampedDistance = Math.max(MIN_PALM_DISTANCE, Math.min(MAX_PALM_DISTANCE, palmDistance));
    
    // Map palm distance to camera distance
    // Larger palm distance (closer hand) → lower camera distance (zoom in)
    // Smaller palm distance (farther hand) → higher camera distance (zoom out)
    const normalized = (clampedDistance - MIN_PALM_DISTANCE) / (MAX_PALM_DISTANCE - MIN_PALM_DISTANCE);
    // Invert: normalized goes from 0 (far) to 1 (close)
    // Camera distance should go from MAX (far) to MIN (close)
    const distance = MAX_CAMERA_DISTANCE - (normalized * (MAX_CAMERA_DISTANCE - MIN_CAMERA_DISTANCE));
    
    return distance;
}

// Detect hand rotation to control camera rotation ONLY (not PC selection)
// Only considers hand rotation in the -90 to -120 degree range and maps it to full 360 degrees
function detectHandRotation(landmarks) {
    if (!landmarks || landmarks.length < 21) {
        return cameraRotationY; // Return current rotation if can't detect
    }
    
    const wrist = landmarks[0];
    const middleMCP = landmarks[9]; // Middle finger MCP joint
    
    // Calculate angle from wrist to middle finger MCP in x-y plane
    const dx = middleMCP.x - wrist.x;
    const dy = middleMCP.y - wrist.y;
    
    // Calculate angle in radians, then convert to degrees
    let handAngleRad = Math.atan2(dy, dx);
    let handAngleDeg = handAngleRad * 180 / Math.PI;
    
    // Normalize to -180 to 180 range
    while (handAngleDeg > 180) handAngleDeg -= 360;
    while (handAngleDeg < -180) handAngleDeg += 360;
    
    // Store raw hand angle for display
    currentHandAngleDeg = handAngleDeg;
    
    // Define the hand rotation range we want to use: -90 to -120 degrees
    const HAND_MIN_DEG = -120;
    const HAND_MAX_DEG = -60;
    
    // Clamp the hand angle to the valid range
    let clampedHandAngle = Math.max(HAND_MIN_DEG, Math.min(HAND_MAX_DEG, handAngleDeg));
    
    // Map the clamped hand angle from [-120, -90] to [0, 2π] for camera rotation
    // When hand is at -90 degrees → camera rotation = 0
    // When hand is at -120 degrees → camera rotation = 2π (360 degrees)
    const normalized = (clampedHandAngle - HAND_MIN_DEG) / (HAND_MAX_DEG - HAND_MIN_DEG);
    let cameraAngle = normalized * 2 * Math.PI;
    
    return cameraAngle;
}

function distance3D(a, b) {
    return Math.sqrt(
        Math.pow(a.x - b.x, 2) +
        Math.pow(a.y - b.y, 2) +
        Math.pow(a.z - b.z, 2)
    );
}

// Update UI
function updateStatus(text) {
    document.getElementById('status-text').textContent = text;
}

function updateModeText(text) {
    // Update mode text with comprehensive information
    const modeTextElement = document.getElementById('mode-text');
    if (modeTextElement) {
        const fingersText = `Fingers: ${fingerCount}`;
        const distanceText = `Distance: ${currentHandDistance.toFixed(3)}`;
        const angleText = `Angle: ${currentHandAngleDeg.toFixed(1)}°`;
        modeTextElement.textContent = `Mode: ${text} | ${fingersText} | ${distanceText} | ${angleText}`;
    }
}

// Update legend with cell types and their colors
function updateLegend(celltype_colors) {
    const legendContent = document.getElementById('legend-content');
    if (!legendContent || !celltype_colors) {
        return;
    }
    
    // Clear existing legend items
    legendContent.innerHTML = '';
    
    // Get sorted list of cell types
    const celltypes = Object.keys(celltype_colors).sort();
    
    // Create legend items
    celltypes.forEach(celltype => {
        const colorInfo = celltype_colors[celltype];
        if (!colorInfo || !colorInfo.hex) return;
        
        // Create legend item
        const item = document.createElement('div');
        item.className = 'legend-item';
        
        // Create color square
        const colorSquare = document.createElement('div');
        colorSquare.className = 'legend-color';
        colorSquare.style.backgroundColor = colorInfo.hex;
        
        // Create label
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = celltype;
        
        // Assemble item
        item.appendChild(colorSquare);
        item.appendChild(label);
        
        // Add to legend
        legendContent.appendChild(item);
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Smoothly interpolate camera rotation towards target
    // Handle angle wrapping (shortest path around circle)
    let angleDiff = targetRotationY - cameraRotationY;
    // Normalize to -π to π range for shortest path
    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
    
    // Smooth interpolation (lerp)
    cameraRotationY += angleDiff * ROTATION_SMOOTHING;
    
    // Smoothly interpolate camera distance (zoom) towards target
    const distanceDiff = targetCameraDistance - cameraDistance;
    cameraDistance += distanceDiff * ZOOM_SMOOTHING;
    
    // Update current rotation angle for UI display
    currentRotationAngle = (cameraRotationY * 180 / Math.PI);
    
    // Update mode text every frame to show real-time hand detection data
    const modeTextElement = document.getElementById('mode-text');
    if (modeTextElement) {
        // Extract mode name from current text or use current mode
        const currentText = modeTextElement.textContent;
        const modeMatch = currentText.match(/^Mode: (.+?)( \| |$)/);
        const modeName = modeMatch ? modeMatch[1] : currentMode;
        
        const fingersText = `Fingers: ${fingerCount}`;
        const distanceText = `Distance: ${currentHandDistance.toFixed(3)}`;
        const angleText = `Angle: ${currentHandAngleDeg.toFixed(1)}°`;
        modeTextElement.textContent = `Mode: ${modeName} | ${fingersText} | ${distanceText} | ${angleText}`;
    }
    
    updateCameraPosition();
    
    // Handle transition animation
    if (isAnimating && sourcePositions && targetPositions) {
        const currentTime = performance.now();
        const elapsed = currentTime - animationStartTime;
        const progress = Math.min(elapsed / animationDuration, 1.0);
        
        // Use ease-in-out for smooth animation
        const easedProgress = progress < 0.5
            ? 2 * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 2) / 2;
        
        // Interpolate between source and target positions
        const interpolatedPositions = sourcePositions.map((src, i) => {
            const tgt = targetPositions[i] || [0, 0, 0];
            return [
                src[0] + (tgt[0] - src[0]) * easedProgress,
                src[1] + (tgt[1] - src[1]) * easedProgress,
                src[2] + (tgt[2] - src[2]) * easedProgress
            ];
        });
        
        updateAnimationPositions(interpolatedPositions);
        
        // Check if animation is complete
        if (progress >= 1.0) {
            isAnimating = false;
            setPositions(targetPositions); // Ensure we end at exact target positions
        }
    }
    
    renderer.render(scene, camera);
}

// Initialize everything
async function init() {
    initScene();
    await loadEmbeddings();
    await initHandDetection();
    animate();
}

// Start the app
init();
