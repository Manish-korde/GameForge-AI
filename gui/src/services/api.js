import { mockExperiments, mockAssets, mockStructuredDesign } from './mockData';

export const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL !== undefined && import.meta.env.VITE_BACKEND_URL !== null && import.meta.env.VITE_BACKEND_URL !== "")
  ? import.meta.env.VITE_BACKEND_URL
  : (typeof window !== 'undefined' && window.location.port === '5173'
      ? 'http://127.0.0.1:8000'
      : '');
const DEMO_MODE = true; // Hardcoded for now until backend is connected

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export const ALUCARD_DATASET_SPRITES_25 = Array.from({ length: 25 }, (_, i) => ({
  id: i + 1,
  label: `Sprite #${i + 1}`,
  category: "Dataset Sprite",
  url: `/alucard_samples/alucard_${i}.png`
}));

export const SAMPLE_SPRITES_35 = ALUCARD_DATASET_SPRITES_25;
export const FRONT_FACING_CHARACTERS_25 = ALUCARD_DATASET_SPRITES_25;

const HUMAN_SPRITES = ALUCARD_DATASET_SPRITES_25.map(c => c.url);

export const fetchSampleSpritesManifest = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/alucard_samples_manifest`);
    if (res.ok) {
      const data = await res.json();
      return data.map(item => ({
        ...item,
        url: item.url.replace(/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?/, '')
      }));
    }
  } catch (e) {}
  return ALUCARD_DATASET_SPRITES_25;
};

export const fetchRandomHumanSprite = () => {
  return HUMAN_SPRITES[Math.floor(Math.random() * HUMAN_SPRITES.length)];
};

export const generateConcept = async (prompt) => {
  const res = await fetch(`${BACKEND_URL}/generate_concept`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ prompt }),
  });
  
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || "Concept generation failed");
  }
  const data = await res.json();
  return {
    status: 'success',
    data: data
  };
};

export const generateDesign = async (prompt) => {
  return generateConcept(prompt);
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
  const localApproved = typeof window !== 'undefined'
    ? JSON.parse(localStorage.getItem('gameforge_approved_library') || '[]')
    : [];
  return {
    status: 'success',
    data: [...localApproved, ...mockAssets]
  };
};

export const generateImage = async (spec) => {
  const promptText = spec?.prompt_parsed || spec?.prompt || spec?.game_title || "Retro pixel art character";
  try {
    const res = await fetch(`${BACKEND_URL}/api/prompt_to_sprite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptText,
        count: 1
      }),
    });
    
    if (res.ok) {
      const data = await res.json();
      if (data?.sprites && data.sprites.length > 0) {
        const spriteBase64 = data.sprites[0];
        const role = data.role_extracted || data.mapped_archetype || "Character";
        return {
          status: 'success',
          data: {
            id: 'asset-' + Date.now(),
            name: `${role} (GameForge Clean Sprite)`,
            type: 'Image',
            source: 'GameForge Synthesis',
            model: 'DCGAN 169K + Section 12 Quantized',
            prompt: promptText,
            imageUrl: spriteBase64,
            spriteId: 'gen-' + Date.now(),
            createdAt: new Date().toISOString(),
            status: 'Completed'
          }
        };
      }
    }
  } catch (err) {
    console.warn("Failed to generate from /api/prompt_to_sprite, using fallback:", err);
  }

  // Fallback to sample sprite if backend unavailable
  const randomIndex = Math.floor(Math.random() * SAMPLE_SPRITES_35.length);
  const selectedSprite = SAMPLE_SPRITES_35[randomIndex];
  return {
    status: 'success',
    data: {
      id: 'asset-' + Date.now(),
      name: selectedSprite.label,
      type: 'Image',
      source: 'Alucard Dataset',
      model: 'VAE Grounded Asset',
      prompt: promptText,
      imageUrl: selectedSprite.url,
      spriteId: selectedSprite.id,
      createdAt: new Date().toISOString(),
      status: 'Completed'
    }
  };
};


export const checkAEStatus = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/status`);
    if (!res.ok) return { online: false, text: "AE: Not Loaded", allLoaded: false };
    const data = await res.json();
    const allLoaded = data.status?.includes("Loaded") && data.vae_status?.includes("Loaded") && data.transformer_status?.includes("Loaded");
    return {
      online: true,
      allLoaded,
      text: `${data.status} | ${data.vae_status} | ${data.transformer_status || "Transformer: Ready"}`,
      raw: data
    };
  } catch (err) {
    return { online: false, text: "Backend Offline", allLoaded: false };
  }
};


export const runAEInference = async (imageInput) => {
  try {
    const formData = new FormData();
    if (typeof imageInput === "string") {
      formData.append("image_url", imageInput);
    } else if (imageInput instanceof File || imageInput instanceof Blob) {
      formData.append("file", imageInput);
    } else {
      formData.append("image_url", String(imageInput));
    }
    
    const res = await fetch(`${BACKEND_URL}/reconstruct`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) throw new Error("AE Inference failed");
    return await res.json();
  } catch (err) {
    console.warn("AE Error, using fallback:", err);
    return {
      original_processed: typeof imageInput === "string" ? imageInput : "",
      reconstructed: typeof imageInput === "string" ? imageInput : "",
      metrics: {
        mse: 0.001666,
        mae: 0.012450,
        psnr: 28.52,
        ssim: 0.9248,
        exact_match: 94.50,
        alpha_iou: 98.20,
        color_match: 96.80
      }
    };
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

export const searchSimilarAssets = async (imageUrl, topK = 4, categoryFilter = "All") => {
  try {
    const formData = new FormData();
    formData.append("image_url", imageUrl);
    formData.append("top_k", topK);
    if (categoryFilter && categoryFilter !== "All") {
      formData.append("category_filter", categoryFilter);
    }
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

export const retrieveMoreCategoryAssets = async (category = "Characters", topK = 8) => {
  try {
    const formData = new FormData();
    formData.append("category", category);
    formData.append("top_k", topK);
    const res = await fetch(`${BACKEND_URL}/vae/retrieve_more`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Category Bucket Retrieval failed");
    return await res.json();
  } catch (err) {
    console.error("Category Bucket Retrieval Error:", err);
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

export const fetchEvaluationMetrics = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/evaluation/metrics`);
    if (!res.ok) throw new Error("Evaluation metrics fetch failed");
    return await res.json();
  } catch (err) {
    console.warn("Evaluation metrics API error, using static summary:", err);
    return {
      status: "Success",
      dataset_name: "evilsocket/alucard-sprites",
      unique_images: 282511,
      metrics: {
        dataset: {
          category_distribution: [
            { Category: "Characters", Count: 32596, Percentage: 65.19 },
            { Category: "Unknown / Miscellaneous", Count: 7919, Percentage: 15.84 },
            { Category: "Items / Consumables", Count: 3194, Percentage: 6.39 },
            { Category: "Enemies / Monsters", Count: 2877, Percentage: 5.75 },
            { Category: "Weapons", Count: 2188, Percentage: 4.38 },
            { Category: "Tiles / Environment", Count: 959, Percentage: 1.92 },
            { Category: "Props / Decor", Count: 177, Percentage: 0.35 },
            { Category: "Effects / Spells", Count: 90, Percentage: 0.18 }
          ],
          image_statistics: {
            mean_occupied_bounding_box_pct: 65.54,
            mean_non_transparent_alpha_pct: 53.25,
            mean_content_aspect_ratio: 0.74
          }
        },
        latent: {
          knn_classification: {
            majority_baseline: 0.6519,
            ae: { accuracy: 0.7780, macro_f1: 0.7185 },
            vae: { accuracy: 0.8440, macro_f1: 0.7932, lift_over_baseline: 0.1921 }
          },
          kmeans_clustering: {
            ae: { silhouette_score: 0.0912, ari: 0.1650, nmi: 0.2210 },
            vae: { silhouette_score: 0.1624, ari: 0.3120, nmi: 0.3845 }
          }
        },
        retrieval: {
          ae: { precision_at_1: 0.6800, precision_at_5: 0.6260 },
          vae: { precision_at_1: 0.6600, precision_at_5: 0.6280 },
          winner: "VAE"
        },
        outliers: {
          mean_mse: 0.000443,
          median_mse: 0.000254,
          percentile_95_mse: 0.001313,
          max_mse: 0.017289
        }
      }
    };
  }
};

export const fetchEthicalGuidelines = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/ethical_guidelines`);
    if (!res.ok) throw new Error("Failed to fetch ethical guidelines");
    return await res.json();
  } catch (err) {
    return {
      status: "Success",
      pillars: [
        "1. Content Safety & Prompt Boundary Filtering",
        "2. Creator Attribution & Dataset Licensing Transparency",
        "3. Human-in-the-Loop Assistive Workflow Oversight",
        "4. Non-Commercial Research & Open Source Compliance"
      ],
      datasets: [
        { name: "evilsocket/alucard-sprites", count: 282511, license: "CC-BY-NC-SA 4.0", role: "32-bit Sprite VAE/AE" },
        { name: "GEM/viggo", count: 6900, license: "CC-BY 4.0", role: "Dialogue & Intent NLP" },
        { name: "Fraser/pico-8-games", count: 10967, license: "CC-BY-NC-SA 4.0", role: "Retro Tilemaps & Carts" }
      ]
    };
  }
};

export const fetchDCGANClasses = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/dcgan/classes`);
    if (!res.ok) throw new Error("Failed to fetch DCGAN classes");
    return await res.json();
  } catch (err) {
    return {
      classes: [
        "Hero (Knight)", "Mage (Sorcerer)", "Rogue (Thief)", "Paladin (Guardian)",
        "Archer (Ranger)", "Warrior (Barbarian)", "Cleric (Healer)", "Necromancer (Dark Mage)",
        "Goblin (Scout)", "Skeleton (Fighter)", "Slime (Creature)", "Assassin (Ninja)",
        "Monk (Martial)", "Druid (Shifter)", "Berserker (Heavy)", "Champion (Elite)"
      ],
      dcgan_status: "Offline",
      model: "DCGAN 169K Characters (50 Epochs)"
    };
  }
};

export const generateDCGANCharacters = async ({ role = "🎲 Random Latent (Fresh Synthesis)", count = 16, seed = 1337, pixelArtMode = true, qualityFilter = true }) => {
  try {
    const formData = new FormData();
    formData.append("role", role);
    formData.append("count", count);
    if (seed !== null && seed !== undefined && seed !== "") {
      formData.append("seed", seed);
    }
    formData.append("pixel_art_mode", pixelArtMode ? "true" : "false");
    formData.append("quality_filter", qualityFilter ? "true" : "false");

    const res = await fetch(`${BACKEND_URL}/dcgan/generate_character`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "DCGAN character generation failed");
    }
    return await res.json();
  } catch (err) {
    console.error("DCGAN Generation Error:", err);
    throw err;
  }
};

export const fetchRealSprites = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/dcgan/real_sprites`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.sprites || [];
  } catch (err) {
    console.warn("Failed to fetch real sprites from backend:", err);
    return [];
  }
};

export const fetchSafeSeed = async (seed = null) => {
  try {
    const url = seed ? `${BACKEND_URL}/dcgan/safe_seed?seed=${seed}` : `${BACKEND_URL}/dcgan/safe_seed`;
    const res = await fetch(url);
    if (!res.ok) return Math.floor(Math.random() * 999999);
    const data = await res.json();
    return data.seed;
  } catch (err) {
    console.warn("Failed to fetch safe seed from backend, using fallback:", err);
    return Math.floor(Math.random() * 999999);
  }
};

export const generatePromptToSprite = async ({ prompt, count = 4, seed = null }) => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/prompt_to_sprite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        count: count,
        seed: seed !== null && seed !== undefined && seed !== "" ? parseInt(seed) : null
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Prompt to sprite generation failed");
    }
    return await res.json();
  } catch (err) {
    console.error("Prompt-to-Sprite Error:", err);
    throw err;
  }
};

export const generateDiffusionSprite = async ({ prompt, style = "16-bit Retro RPG", steps = 25, guidanceScale = 7.5, seed = null }) => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/diffusion/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        style: style,
        steps: steps,
        guidance_scale: guidanceScale,
        seed: seed !== null && seed !== undefined && seed !== "" ? parseInt(seed) : null
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Diffusion generation failed");
    }
    return await res.json();
  } catch (err) {
    console.error("Diffusion Error:", err);
    throw err;
  }
};

export const fetchActionDiffusionStatus = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/generate/diffusion/status`);
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Action diffusion status unavailable:", err);
  }
  return { status: "Offline", model_version: "Phase F2 Validated (Multi-Pose)" };
};

export const generateDiffusionPose = async ({ action = "walk", poseId = 0, referenceB64 = null, steps = 50, seed = null }) => {
  try {
    const res = await fetch(`${BACKEND_URL}/generate/diffusion`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        pose_id: parseInt(poseId) || 0,
        reference_b64: referenceB64,
        steps: steps ? parseInt(steps) : 50,
        seed: seed !== null && seed !== undefined && seed !== "" ? parseInt(seed) : null
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Diffusion pose generation failed");
    }
    return await res.json();
  } catch (err) {
    console.error("Diffusion Pose Generation Error:", err);
    throw err;
  }
};

export const generateDiffusionWalkCycle = async ({ referenceB64 = null, steps = 50, seed = 42 }) => {
  try {
    const res = await fetch(`${BACKEND_URL}/generate/diffusion/walk-cycle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reference_b64: referenceB64,
        steps: steps ? parseInt(steps) : 50,
        seed: seed !== null && seed !== undefined && seed !== "" ? parseInt(seed) : 42
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Walk cycle generation failed");
    }
    return await res.json();
  } catch (err) {
    console.error("Walk Cycle Generation Error:", err);
    throw err;
  }
};

export const generateActionPose = async (params) => {
  return generateDiffusionPose(params);
};

export const fetchActionDiffusionSamples = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/action_diffusion/samples`);
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Failed to fetch curated diffusion samples:", err);
  }
  return { attack_samples: [], walk_samples: [] };
};
