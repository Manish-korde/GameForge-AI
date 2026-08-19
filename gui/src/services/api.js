import { mockExperiments, mockAssets, mockStructuredDesign } from './mockData';

const DEMO_MODE = true; // Hardcoded for now until backend is connected

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export const generateDesign = async (prompt) => {
  if (DEMO_MODE) {
    await delay(1500); // simulate processing
    return {
      status: 'success',
      data: mockStructuredDesign(prompt)
    };
  }
  // Future: Real API call
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
        // using PokeAPI for actual 2D pixel art sprites instead of random photos
        imageUrl: `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${Math.floor(Math.random() * 898) + 1}.png`,
        createdAt: new Date().toISOString(),
        status: 'Completed'
      }
    };
  }
};

const BACKEND_URL = "http://localhost:8000";

export const checkAEStatus = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/status`);
    const data = await res.json();
    return data.status;
  } catch (err) {
    return "Autoencoder: Model unavailable";
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
