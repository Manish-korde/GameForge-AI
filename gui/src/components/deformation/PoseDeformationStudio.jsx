import React, { useState, useEffect } from 'react';
import PoseSelector from './PoseSelector';
import JointEditor from './JointEditor';
import DeformationControls from './DeformationControls';
import DeformationPreview from './DeformationPreview';
import AnimationTimeline from './AnimationTimeline';
import { fetchPoseTemplates, detectCharacterSkeleton, deformCharacterPose } from '../../services/deformationApi';
import '../PoseDeformationStudio.css';

/**
 * PoseDeformationStudio: modular container coordinating template selection,
 * interactive joint editing, deformation dispatch, animation generation, and spritesheet export.
 */
const PoseDeformationStudio = ({ imageUrl, assetName, onBack }) => {
  // Studio Mode: 'pose' (Single Pose Editor) | 'animation' (Multi-Frame Animation Studio)
  const [studioMode, setStudioMode] = useState('pose');

  // Pose templates & selection
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState('walk_1');
  const [intensity, setIntensity] = useState(1.0);
  const [method, setMethod] = useState('geometric'); // 'geometric' | 'hybrid'
  const [resampling, setResampling] = useState('nearest'); // 'nearest' | 'bilinear'
  const [preservePixelArt, setPreservePixelArt] = useState(true);

  // Skeletons
  const [sourceSkeleton, setSourceSkeleton] = useState(null);
  const [targetSkeleton, setTargetSkeleton] = useState(null);
  const [isSkeletonVisible, setIsSkeletonVisible] = useState(true);
  const [activeJoint, setActiveJoint] = useState(null);

  // Results & UI State
  const [isLoading, setIsLoading] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [viewMode, setViewMode] = useState('side_by_side'); // 'side_by_side' | 'ghost_overlay'

  // 1. Load Templates on mount
  useEffect(() => {
    let isMounted = true;
    fetchPoseTemplates()
      .then(data => {
        if (isMounted && data && data.templates) {
          setTemplates(data.templates);
          if (data.templates.length > 0) {
            setSelectedTemplate(data.templates[1]?.id || data.templates[0].id);
          }
        }
      })
      .catch(err => {
        console.warn('Could not load pose templates from backend:', err);
      });
    return () => { isMounted = false; };
  }, []);

  // 2. Auto-detect Skeleton on mount or when imageUrl changes
  useEffect(() => {
    if (!imageUrl) return;
    handleAutoDetectSkeleton();
  }, [imageUrl]);

  const handleAutoDetectSkeleton = async () => {
    setIsDetecting(true);
    setError(null);
    try {
      const payload = imageUrl.startsWith('data:') 
        ? { image_b64: imageUrl } 
        : { image_url: imageUrl };

      const res = await detectCharacterSkeleton(payload);
      if (res && res.skeleton) {
        setSourceSkeleton(res.skeleton);
      }
    } catch (err) {
      console.warn('Auto-detect skeleton failed:', err);
      setError('Could not auto-detect skeleton. Using standard anatomical coordinates.');
    } finally {
      setIsDetecting(false);
    }
  };

  // 3. Handle Joint Position Update from JointEditor
  const handleJointChange = (jointName, clampedX, clampedY) => {
    const currentSkel = targetSkeleton || sourceSkeleton;
    if (!currentSkel || !currentSkel.joints) return;

    let updatedJoints;
    if (Array.isArray(currentSkel.joints)) {
      updatedJoints = currentSkel.joints.map(j => {
        if (j.name === jointName) {
          return { ...j, x: clampedX, y: clampedY };
        }
        return j;
      });
    } else {
      // Canonical dictionary representation
      const existing = currentSkel.joints[jointName] || {};
      updatedJoints = {
        ...currentSkel.joints,
        [jointName]: {
          ...existing,
          name: jointName,
          x: clampedX,
          y: clampedY,
          is_visible: existing.is_visible !== undefined ? existing.is_visible : true,
          confidence: existing.confidence !== undefined ? existing.confidence : 1.0
        }
      };
    }

    setTargetSkeleton({ ...currentSkel, joints: updatedJoints });
  };

  // 4. Run Pose Deformation
  const handleRunDeformation = async () => {
    if (!imageUrl) return;
    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        target_pose: selectedTemplate,
        method: method,
        resampling: resampling,
        intensity: parseFloat(intensity),
        preserve_pixel_art: preservePixelArt,
        maintain_canvas_size: true
      };

      if (imageUrl.startsWith('data:')) {
        payload.image_b64 = imageUrl;
      } else {
        payload.image_url = imageUrl;
      }

      // If user custom-adjusted the skeleton, pass target_skeleton
      if (targetSkeleton && targetSkeleton.joints) {
        payload.target_skeleton = targetSkeleton;
      }

      const res = await deformCharacterPose(payload);
      setResult(res);
      if (res.target_skeleton) {
        setTargetSkeleton(res.target_skeleton);
      }
    } catch (err) {
      console.error('Deformation error:', err);
      setError(err.message || 'Deformation processing failed.');
    } finally {
      setIsLoading(false);
    }
  };

  // 5. Reset target skeleton to template default
  const handleResetPose = () => {
    setTargetSkeleton(null);
    setIntensity(1.0);
    setResult(null);
  };

  // 6. Export Deformed Sprite as Transparent PNG
  const handleExportPNG = () => {
    if (!result || !result.output_image_b64) return;
    const a = document.createElement('a');
    a.href = result.output_image_b64;
    a.download = `${assetName || 'character'}_${selectedTemplate}_deformed.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const currentActiveSkel = targetSkeleton || sourceSkeleton;

  return (
    <div className="pose-studio">
      {/* Studio Header */}
      <div className="pose-studio-header flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xl">🦴</span>
            <h3 className="text-xl font-bold text-white tracking-wide">
              Character Pose Deformation Studio
            </h3>
            <span className="badge bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs px-2 py-0.5 rounded">
              v1.0 Modular Baseline
            </span>
          </div>
          <p className="text-xs text-secondary">
            Modular 2D skeletal linear blend skinning with nearest-neighbor pixel-art resampling and optional β-VAE refinement.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Mode Switcher Tabs */}
          <div className="flex bg-[#121620] border border-[#252b38] rounded-lg p-0.5">
            <button
              className={`text-xs font-semibold px-3 py-1.5 rounded-md transition-all ${
                studioMode === 'pose'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setStudioMode('pose')}
            >
              🦴 Pose Editor
            </button>
            <button
              className={`text-xs font-semibold px-3 py-1.5 rounded-md transition-all ${
                studioMode === 'animation'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-gray-400 hover:text-white'
              }`}
              onClick={() => setStudioMode('animation')}
            >
              🎬 Animation & Spritesheet
            </button>
          </div>

          {onBack && (
            <button className="btn btn-secondary text-xs px-3 py-1.5" onClick={onBack}>
              ← Back to Asset Studio
            </button>
          )}
        </div>
      </div>

      {/* Conditional View: Single Pose Editor vs Multi-Frame Animation Studio */}
      {studioMode === 'animation' ? (
        <AnimationTimeline imageUrl={imageUrl} assetName={assetName} />
      ) : (
        /* Main Grid: Controls Left, Canvases Right */
        <div className="grid grid-cols-12 gap-6">
        {/* Left Column: Pose Templates & Parameters (5 cols) */}
        <div className="col-span-5 flex flex-col gap-4">
          <PoseSelector
            templates={templates}
            selectedTemplate={selectedTemplate}
            onSelectTemplate={(tplId) => {
              setSelectedTemplate(tplId);
              setTargetSkeleton(null); // Reset overrides on template switch
            }}
          />

          <DeformationControls
            intensity={intensity}
            onChangeIntensity={setIntensity}
            method={method}
            onChangeMethod={setMethod}
            resampling={resampling}
            onChangeResampling={setResampling}
            preservePixelArt={preservePixelArt}
            onChangePreservePixelArt={setPreservePixelArt}
            isLoading={isLoading}
            isDetecting={isDetecting}
            onRunDeformation={handleRunDeformation}
            onResetPose={handleResetPose}
          />
        </div>

        {/* Right Column: Previews, Skeleton Overlay & QA Metrics (7 cols) */}
        <div className="col-span-7 flex flex-col gap-4">
          {/* Top Bar: Skeleton Toggle & View Mode */}
          <div className="flex items-center justify-between bg-[#121620] p-2.5 rounded-lg border border-[#252b38]">
            <div className="flex items-center gap-3">
              <button
                className={`text-xs px-2.5 py-1 rounded transition-colors ${
                  isSkeletonVisible
                    ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/40'
                    : 'bg-[#181d28] text-gray-400 hover:text-white'
                }`}
                onClick={() => setIsSkeletonVisible(!isSkeletonVisible)}
              >
                🦴 Skeleton Overlay {isSkeletonVisible ? 'ON' : 'OFF'}
              </button>

              <button
                className="text-xs px-2.5 py-1 rounded bg-[#181d28] text-gray-400 hover:text-white transition-colors"
                onClick={handleAutoDetectSkeleton}
                disabled={isDetecting}
              >
                {isDetecting ? 'Detecting...' : '🎯 Re-detect Joints'}
              </button>
            </div>

            {/* View Mode Selector */}
            <div className="flex gap-1 bg-[#0b0d12] p-0.5 rounded">
              <button
                className={`text-[11px] px-2 py-0.5 rounded ${viewMode === 'side_by_side' ? 'bg-[#252b38] text-white' : 'text-gray-400'}`}
                onClick={() => setViewMode('side_by_side')}
              >
                Side by Side
              </button>
              <button
                className={`text-[11px] px-2 py-0.5 rounded ${viewMode === 'ghost_overlay' ? 'bg-[#252b38] text-white' : 'text-gray-400'}`}
                onClick={() => setViewMode('ghost_overlay')}
              >
                Ghost Overlay
              </button>
            </div>
          </div>

          {/* Interactive Preview Area */}
          <div className="flex gap-4 items-center justify-center bg-[#0e1118] p-4 rounded-xl border border-[#202636]">
            {/* Left Canvas: Source / Interactive Skeleton */}
            <div className="flex flex-col items-center">
              <span className="text-[11px] uppercase font-semibold text-gray-400 mb-1.5 flex items-center gap-1.5">
                <span>Source Pose</span>
                {isSkeletonVisible && (
                  <span className="text-[10px] text-indigo-400 font-normal">(Drag joints to edit)</span>
                )}
              </span>

              <JointEditor
                imageUrl={imageUrl}
                skeleton={currentActiveSkel}
                isSkeletonVisible={isSkeletonVisible}
                activeJoint={activeJoint}
                onSelectJoint={setActiveJoint}
                onJointChange={handleJointChange}
              />
            </div>

            {/* Middle Divider Arrow */}
            <div className="flex flex-col items-center text-gray-500">
              <span className="text-2xl">➔</span>
              <span className="text-[10px] uppercase font-mono mt-1">{method}</span>
            </div>

            {/* Right Canvas: Deformed Result or Placeholder */}
            <DeformationPreview
              imageUrl={imageUrl}
              result={result}
              viewMode={viewMode}
              assetName={assetName}
              onExportPng={handleExportPNG}
            />
          </div>

          {/* Error Banner */}
          {error && (
            <div className="p-3 bg-red-950/40 border border-red-500/40 rounded-lg text-xs text-red-300 flex items-center gap-2">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  );
};

export default PoseDeformationStudio;
