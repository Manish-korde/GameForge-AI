import { mockExperiments, mockAssets, mockStructuredDesign } from './mockData';

const BACKEND_URL = "http://localhost:8000";
const DEMO_MODE = true; // Hardcoded for now until backend is connected

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const HUMAN_SPRITES = Array.from({ length: 20 }, (_, i) => `http://localhost:8000/alucard_samples/alucard_${i}.png`);

export const fetchRandomHumanSprite = () => {
  return HUMAN_SPRITES[Math.floor(Math.random() * HUMAN_SPRITES.length)];
};

export const generateDesign = async (prompt) => {
  try {
    const formData = new FormData();
    formData.append("prompt", prompt);
    
    const res = await fetch(`${BACKEND_URL}/generate_concept`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) throw new Error("Concept generation failed");
    const data = await res.json();
    return {
      status: 'success',
      data: data
    };
  } catch (err) {
    console.error("Design Generation Error:", err);
    throw err;
  }
};

export const fetchExperiments = async () => {
  if (DEMO_MODE) {
    await delay(500);
    return {
      status: 'success',
      data: mockExperiments
    };
  }
};

export const fetchLibraryAssets = async () => {
  if (DEMO_MODE) {
    await delay(500);
    return {
      status: 'success',
      data: mockAssets
    };
  }
};

export const generateImage = async (spec) => {
  if (DEMO_MODE) {
    await delay(3000);
    return {
      status: 'success',
      data: {
        id: 'asset-' + Date.now(),
        name: spec.theme + ' Asset',
        type: 'Image',
        source: 'Diffusion',
        model: 'Diffusion Model',
        prompt: spec.prompt,
        // using predefined human sprites for VAE interpolation compatibility
        imageUrl: fetchRandomHumanSprite(),
        createdAt: new Date().toISOString(),
        status: 'Completed'
      }
    };
  }
};


export const checkAEStatus = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/status`);
    if (!res.ok) return "AE: Not Loaded";
    const data = await res.json();
    return `${data.status} | ${data.vae_status}`;
  } catch (err) {
    return "Models: Offline";
  }
};


export const runAEInference = async (imageUrl) => {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    
    const res = await fetch(`${BACKEND_URL}/reconstruct`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) throw new Error("AE Inference failed");
    return await res.json();
  } catch (err) {
    console.error("AE Error:", err);
    throw err;
  }
};

export const runNoisyAEInference = async (imageUrl, noiseScale = 0.1) => {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    formData.append("noise_scale", noiseScale);
    
    const res = await fetch(`${BACKEND_URL}/reconstruct_noisy`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) throw new Error("Noisy AE Inference failed");
    return await res.json();
  } catch (err) {
    console.error("Noisy AE Error:", err);
    throw err;
  }
};

export const runVAEInference = async (imageUrl, scale = 1.0) => {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    formData.append("scale", scale);
    
    const res = await fetch(`${BACKEND_URL}/generate_variants`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) throw new Error("VAE Inference failed");
    return await res.json();
  } catch (err) {
    console.error("VAE Error:", err);
    throw err;
  }
};

export const runVAEInterpolation = async (imageUrlA, imageUrlB, alpha = 0.5) => {
  try {
    const formData = new FormData();
    formData.append("image_url_a", imageUrlA);
    formData.append("image_url_b", imageUrlB);
    formData.append("alpha", alpha);
    
    const res = await fetch(`${BACKEND_URL}/interpolate_vae`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) throw new Error("VAE Interpolation failed");
    return await res.json();
  } catch (err) {
    console.error("VAE Interpolation Error:", err);
    throw err;
  }
};

export const searchSimilarAssets = async (imageUrl, topK = 4) => {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    formData.append("top_k", topK);
    const res = await fetch(`${BACKEND_URL}/vae/search_similar`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("VAE Similarity Search failed");
    return await res.json();
  } catch (err) {
    console.error("VAE Search Error:", err);
    throw err;
  }
};

export const clusterAssets = async (urlsArray) => {
  try {
    const formData = new FormData();
    formData.append("urls", urlsArray.join(","));
    const res = await fetch(`${BACKEND_URL}/vae/cluster_assets`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("VAE Clustering failed");
    return await res.json();
  } catch (err) {
    console.error("VAE Clustering Error:", err);
    throw err;
  }
};

export const detectDuplicateAsset = async (imageUrlA, imageUrlB) => {
  try {
    const formData = new FormData();
    formData.append("image_url_a", imageUrlA);
    formData.append("image_url_b", imageUrlB);
    const res = await fetch(`${BACKEND_URL}/vae/detect_duplicate`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("VAE Duplicate Detection failed");
    return await res.json();
  } catch (err) {
    console.error("VAE Duplicate Detection Error:", err);
    throw err;
  }
};

export const getAnomalyScore = async (imageUrl) => {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    const res = await fetch(`${BACKEND_URL}/vae/anomaly_score`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("VAE Anomaly Detection failed");
    return await res.json();
  } catch (err) {
    console.error("VAE Anomaly Detection Error:", err);
    throw err;
  }
};

