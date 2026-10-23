/**
 * GameForge AI — Character Pose Deformation API Service
 * Dedicated client methods for pose templates, skeleton detection, and deformation execution.
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL !== undefined 
  ? import.meta.env.VITE_BACKEND_URL 
  : "http://127.0.0.1:8000";

/**
 * Fetches the tested catalog of named pose templates (idle, walk, run, jump, crouch, attack).
 */
export const fetchPoseTemplates = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/deformation/templates`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to load pose templates (${res.status})`);
    }
    const data = await res.json();
    if (!data || !Array.isArray(data.templates)) {
      throw new Error('Malformed templates response payload');
    }
    return data;
  } catch (err) {
    console.error('fetchPoseTemplates error:', err);
    throw err;
  }
};

/**
 * Auto-detects the non-transparent sprite bounding box and aligns standard 16-joint skeleton.
 */
export const detectCharacterSkeleton = async ({ image_b64, image_url }) => {
  try {
    if (!image_b64 && !image_url) {
      throw new Error('Must provide either image_b64 or image_url for skeleton detection');
    }
    const res = await fetch(`${BACKEND_URL}/deformation/detect_skeleton`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_b64, image_url })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Skeleton auto-detection failed (${res.status})`);
    }
    const data = await res.json();
    if (!data || !data.skeleton || !data.skeleton.joints) {
      throw new Error('Invalid skeleton detection response');
    }
    return data;
  } catch (err) {
    console.error('detectCharacterSkeleton error:', err);
    throw err;
  }
};

/**
 * Submits a sprite and target pose definition for 2D skeletal deformation.
 */
export const deformCharacterPose = async (payload) => {
  try {
    if (!payload.image_b64 && !payload.image_url) {
      throw new Error('Deformation payload requires source image (image_b64 or image_url)');
    }
    const res = await fetch(`${BACKEND_URL}/deformation/pose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Deformation request failed (${res.status})`);
    }
    const data = await res.json();
    if (!data.output_image_b64) {
      throw new Error('Deformation response missing output image');
    }
    return data;
  } catch (err) {
    console.error('deformCharacterPose error:', err);
    throw err;
  }
};

/**
 * Fetches the catalog of supported character animation presets (idle, walk, run, jump, attack).
 */
export const fetchAnimationPresets = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/deformation/animation/presets`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to fetch animation presets (${res.status})`);
    }
    const data = await res.json();
    if (!data || !Array.isArray(data.presets)) {
      throw new Error('Malformed animation presets response payload');
    }
    return data;
  } catch (err) {
    console.error('fetchAnimationPresets error:', err);
    throw err;
  }
};

/**
 * Generates an animated multi-frame sequence from a sprite and motion preset.
 */
export const generateCharacterAnimation = async (payload) => {
  try {
    if (!payload.image_b64 && !payload.image_url) {
      throw new Error('Animation payload requires source image (image_b64 or image_url)');
    }
    const res = await fetch(`${BACKEND_URL}/deformation/animate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Animation generation failed (${res.status})`);
    }
    const data = await res.json();
    if (!data.frames || !Array.isArray(data.frames) || data.frames.length === 0) {
      throw new Error('Animation response contained no generated frames');
    }
    return data;
  } catch (err) {
    console.error('generateCharacterAnimation error:', err);
    throw err;
  }
};

/**
 * Packs animation frames into a transparent PNG spritesheet grid with metadata.
 */
export const exportAnimationSpritesheet = async (payload) => {
  try {
    const res = await fetch(`${BACKEND_URL}/deformation/spritesheet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Spritesheet export failed (${res.status})`);
    }
    const data = await res.json();
    if (!data.spritesheet_image_b64 || !data.metadata) {
      throw new Error('Invalid spritesheet response payload');
    }
    return data;
  } catch (err) {
    console.error('exportAnimationSpritesheet error:', err);
    throw err;
  }
};

