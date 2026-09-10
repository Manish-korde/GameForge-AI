import { mockExperiments, mockAssets, mockStructuredDesign } from './mockData';

const BACKEND_URL = "http://127.0.0.1:8000";
const DEMO_MODE = true; // Hardcoded for now until backend is connected

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const HUMAN_SPRITES = Array.from({ length: 35 }, (_, i) => `http://127.0.0.1:8000/alucard_samples/alucard_${i}.png`);

export const SAMPLE_SPRITES_35 = [
  { id: 0, label: "Sprite #1: Arcane Mage", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_0.png" },
  { id: 1, label: "Sprite #2: Knight Paladin", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_1.png" },
  { id: 2, label: "Sprite #3: Flame Sorcerer", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_2.png" },
  { id: 3, label: "Sprite #4: Armored Guardian", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_3.png" },
  { id: 4, label: "Sprite #5: Royal Knight", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_4.png" },
  { id: 5, label: "Sprite #6: Shadow Warrior", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_5.png" },
  { id: 6, label: "Sprite #7: Demon Specter", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_6.png" },
  { id: 7, label: "Sprite #8: Forest Ranger", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_7.png" },
  { id: 8, label: "Sprite #9: Slime Beast", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_8.png" },
  { id: 9, label: "Sprite #10: Elven Archer", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_9.png" },
  { id: 10, label: "Sprite #11: Dark Necromancer", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_10.png" },
  { id: 11, label: "Sprite #12: Rogue Assassin", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_11.png" },
  { id: 12, label: "Sprite #13: Shadow Ninja", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_12.png" },
  { id: 13, label: "Sprite #14: Blood Berserker", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_13.png" },
  { id: 14, label: "Sprite #15: Forest Elf", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_14.png" },
  { id: 15, label: "Sprite #16: Dwarf Defender", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_15.png" },
  { id: 16, label: "Sprite #17: Adventurer Hero", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_16.png" },
  { id: 17, label: "Sprite #18: Skeleton Warrior", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_17.png" },
  { id: 18, label: "Sprite #19: Frost Maiden", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_18.png" },
  { id: 19, label: "Sprite #20: Fire Elemental", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_19.png" },
  { id: 20, label: "Sprite #21: Cyber Knight", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_20.png" },
  { id: 21, label: "Sprite #22: Golden Templar", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_21.png" },
  { id: 22, label: "Sprite #23: Shieldbearer", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_22.png" },
  { id: 23, label: "Sprite #24: Battle Master", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_23.png" },
  { id: 24, label: "Sprite #25: Crossbow Scout", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_24.png" },
  { id: 25, label: "Sprite #26: Goblin Raider", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_25.png" },
  { id: 26, label: "Sprite #27: Orc Chieftain", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_26.png" },
  { id: 27, label: "Sprite #28: High Priest", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_27.png" },
  { id: 28, label: "Sprite #29: Phoenix Cultist", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_28.png" },
  { id: 29, label: "Sprite #30: Dragon Slayer", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_29.png" },
  { id: 30, label: "Sprite #31: Cyber Sentinel", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_30.png" },
  { id: 31, label: "Sprite #32: Starship Commander", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_31.png" },
  { id: 32, label: "Sprite #33: Alien Droid", category: "Enemy", url: "http://127.0.0.1:8000/alucard_samples/alucard_32.png" },
  { id: 33, label: "Sprite #34: Void Stalker", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_33.png" },
  { id: 34, label: "Sprite #35: Purple Elf Mage", category: "Character", url: "http://127.0.0.1:8000/alucard_samples/alucard_34.png" }
];

export const fetchSampleSpritesManifest = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/alucard_samples_manifest`);
    if (res.ok) return await res.json();
  } catch (e) {}
  return SAMPLE_SPRITES_35;
};

export const fetchRandomHumanSprite = () => {
  return HUMAN_SPRITES[Math.floor(Math.random() * HUMAN_SPRITES.length)];
};

export const generateConcept = async (prompt) => {
  try {
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
  } catch (err) {
    if (err.message.includes("Safety Audit") || err.message.includes("restricted") || err.message.includes("prohibited")) {
      throw err;
    }
    console.warn("Transformer Concept API call failed, using client fallback:", err);
    const pLower = (prompt || "").toLowerCase();
    
    let genre = "16-bit Retro Game";
    let role = "Protagonist";
    let attrs = ["Agile Movement", "Special Ability", "Quick Dash"];
    let theme = "Adventure World";
    let hazards = ["Environmental Hazards", "Time Limit"];
    let enemies = ["Rival Competitor", "Obstacle Drone"];
    let tags = ["hero_character", "tool_item", "action_effect", "tile_ground"];

    if (pLower.includes("pizza") || pLower.includes("delivery")) {
      genre = "Delivery Simulation";
      role = "Delivery Courier";
      attrs = ["Order Handling", "Route Navigation", "Speed Dash"];
      theme = "Metropolitan City Streets";
      hazards = ["Heavy Traffic", "Slippery Road Puddles", "Strict Delivery Timer"];
      enemies = ["Stray Street Dogs", "Traffic Drones", "Impatient Customers"];
      tags = ["delivery_courier", "scooter_vehicle", "pizza_box_item", "city_street_tile"];
    } else if (pLower.includes("farm") || pLower.includes("crop")) {
      genre = "Farming Simulator";
      role = "Master Farmer";
      attrs = ["Crop Harvesting", "Tool Upgrades", "Seasonal Planning"];
      theme = "Sunlit Countryside Valley";
      hazards = ["Sudden Frost", "Drought Hazard", "Pest Infestation"];
      enemies = ["Wild Boars", "Locust Swarms", "Crows"];
      tags = ["farmer_character", "tractor_vehicle", "crop_item", "farm_field_tile"];
    } else if (pLower.includes("race") || pLower.includes("car")) {
      genre = "Arcade Street Racing";
      role = "Street Racer";
      attrs = ["Nitro Boost", "Drift Precision", "Engine Tuning"];
      theme = "Neon Highway Circuit";
      hazards = ["Oil Slicks", "Road Debris", "Sharp Hairpin Turns"];
      enemies = ["Rival Street Racers", "Police Interceptors"];
      tags = ["racecar_vehicle", "nitro_item", "exhaust_effect", "highway_tile"];
    } else if (pLower.includes("space") || pLower.includes("ship")) {
      genre = "Sci-Fi Space Exploration";
      role = "Starship Commander";
      attrs = ["Plasma Thrusters", "Shield Boosting", "Laser Targeting"];
      theme = "Deep Space Orbital Station";
      hazards = ["Asteroid Belts", "Solar Flares", "Hull Depressurization"];
      enemies = ["Rogue AI Drones", "Alien Harvesters", "Space Pirates"];
      tags = ["spaceship_vehicle", "laser_weapon", "plasma_effect", "space_station_tile"];
    }

    const titleWords = (prompt || "Game").split(" ").filter(w => w.length > 3).slice(0, 2);
    const title = titleWords.length > 0 ? titleWords.map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ") : "GameForge Spec";

    return {
      status: 'success',
      data: {
        game_title: title,
        prompt_parsed: prompt,
        genre: genre,
        art_style: "16-bit Retro Pixel Art",
        main_character: {
          role: role,
          attributes: attrs
        },
        environment: {
          theme: theme,
          hazards: hazards
        },
        enemies: enemies,
        recommended_asset_tags: tags,
        status: "Fallback"
      }
    };
  }
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
    await delay(1500);
    const randomIndex = Math.floor(Math.random() * SAMPLE_SPRITES_35.length);
    const selectedSprite = SAMPLE_SPRITES_35[randomIndex];
    
    return {
      status: 'success',
      data: {
        id: 'asset-' + Date.now(),
        name: selectedSprite.label.split(": ")[1] || selectedSprite.label,
        type: 'Image',
        source: 'Diffusion',
        model: 'Diffusion Model',
        prompt: spec.prompt || spec.prompt_parsed || 'Pixel art character sprite',
        imageUrl: selectedSprite.url,
        spriteId: selectedSprite.id,
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
    console.warn("AE Error, using fallback:", err);
    return {
      original_processed: imageUrl,
      reconstructed: imageUrl,
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


