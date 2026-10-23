import React, { useState, useEffect, useRef } from 'react';
import {
  fetchAnimationPresets,
  generateCharacterAnimation,
  exportAnimationSpritesheet
} from '../../services/deformationApi';

/**
 * AnimationTimeline: Interactive 2D character animation player, scrubber,
 * frame filmstrip, and spritesheet export panel.
 */
const AnimationTimeline = ({ imageUrl, assetName }) => {
  // Presets & Parameters
  const [presets, setPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState('walk');
  const [frameCount, setFrameCount] = useState(8);
  const [fps, setFps] = useState(10);
  const [isLooping, setIsLooping] = useState(true);

  // Playback & Frames
  const [animationResult, setAnimationResult] = useState(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Loading & Diagnostics
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExportingSheet, setIsExportingSheet] = useState(false);
  const [error, setError] = useState(null);
  const [spritesheetResult, setSpritesheetResult] = useState(null);
  const [sheetColumns, setSheetColumns] = useState(4);

  const playbackTimerRef = useRef(null);

  // 1. Fetch available animation presets on mount
  useEffect(() => {
    let isMounted = true;
    fetchAnimationPresets()
      .then(data => {
        if (isMounted && data && Array.isArray(data.presets)) {
          setPresets(data.presets);
          const defaultPreset = data.presets.find(p => p.id === 'walk') || data.presets[0];
          if (defaultPreset) {
            setSelectedPresetId(defaultPreset.id);
            setFrameCount(defaultPreset.default_frame_count);
            setFps(defaultPreset.default_fps);
            setIsLooping(defaultPreset.is_looping);
          }
        }
      })
      .catch(err => {
        console.warn('Could not load animation presets:', err);
      });
    return () => { isMounted = false; };
  }, []);

  // 2. Preset change handler: update defaults
  const handleSelectPreset = (presetId) => {
    setSelectedPresetId(presetId);
    const p = presets.find(item => item.id === presetId);
    if (p) {
      setFrameCount(p.default_frame_count);
      setFps(p.default_fps);
      setIsLooping(p.is_looping);
    }
  };

  // 3. Playback animation loop
  useEffect(() => {
    if (!isPlaying || !animationResult || !animationResult.frames || animationResult.frames.length <= 1) {
      if (playbackTimerRef.current) {
        clearInterval(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
      return;
    }

    const intervalMs = Math.max(25, Math.round(1000 / fps));
    playbackTimerRef.current = setInterval(() => {
      setCurrentFrameIdx(prev => {
        const next = prev + 1;
        if (next >= animationResult.frames.length) {
          if (isLooping) {
            return 0;
          } else {
            setIsPlaying(false);
            return prev;
          }
        }
        return next;
      });
    }, intervalMs);

    return () => {
      if (playbackTimerRef.current) {
        clearInterval(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
    };
  }, [isPlaying, fps, animationResult, isLooping]);

  // 4. Generate Animation
  const handleGenerateAnimation = async () => {
    if (!imageUrl) {
      setError('Please provide a source sprite to generate animations.');
      return;
    }

    setIsGenerating(true);
    setIsPlaying(false);
    setError(null);
    setSpritesheetResult(null);

    try {
      const payload = {
        preset_id: selectedPresetId,
        frame_count: parseInt(frameCount, 10),
        fps: parseInt(fps, 10),
        is_looping: isLooping,
        method: 'geometric',
        preserve_pixel_art: true,
        maintain_canvas_size: true
      };

      if (imageUrl.startsWith('data:')) {
        payload.image_b64 = imageUrl;
      } else {
        payload.image_url = imageUrl;
      }

      const res = await generateCharacterAnimation(payload);
      setAnimationResult(res);
      setCurrentFrameIdx(0);
      setIsPlaying(true);
    } catch (err) {
      console.error('Animation generation error:', err);
      setError(err.message || 'Animation generation failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  // 5. Export Spritesheet
  const handleExportSpritesheet = async () => {
    if (!animationResult || !animationResult.frames) {
      setError('Generate an animation first before exporting a spritesheet.');
      return;
    }

    setIsExportingSheet(true);
    setError(null);

    try {
      // Pack the actual generated frames directly
      const framesB64 = animationResult.frames.map(f => f.image_b64);
      const payload = {
        custom_frames_b64: framesB64,
        columns: parseInt(sheetColumns, 10),
        fps: parseInt(fps, 10)
      };

      const res = await exportAnimationSpritesheet(payload);
      setSpritesheetResult(res);
    } catch (err) {
      console.error('Spritesheet export error:', err);
      setError(err.message || 'Spritesheet packaging failed.');
    } finally {
      setIsExportingSheet(false);
    }
  };

  // Download Spritesheet PNG
  const handleDownloadSheet = () => {
    if (!spritesheetResult || !spritesheetResult.spritesheet_image_b64) return;
    const a = document.createElement('a');
    a.href = spritesheetResult.spritesheet_image_b64;
    a.download = `${assetName || 'character'}_${selectedPresetId}_spritesheet.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const frames = animationResult?.frames || [];
  const activeFrame = frames[currentFrameIdx] || null;

  return (
    <div className="flex flex-col gap-6 text-left">
      {/* Top Config Row: Preset Selection & Parameters */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Column: Preset Cards */}
        <div className="col-span-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase font-bold text-gray-400 tracking-wider">
              Motion Presets
            </span>
            <span className="text-[11px] text-gray-500">
              {presets.length} presets available
            </span>
          </div>

          <div className="grid grid-cols-1 gap-2">
            {presets.map(p => {
              const isSelected = selectedPresetId === p.id;
              return (
                <div
                  key={p.id}
                  className={`template-card flex items-center justify-between ${isSelected ? 'active' : ''}`}
                  onClick={() => handleSelectPreset(p.id)}
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-white">{p.name}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-300 uppercase">
                        {p.category}
                      </span>
                    </div>
                    <p className="text-[11px] text-gray-400 mt-0.5 line-clamp-1">
                      {p.description}
                    </p>
                  </div>
                  <div className="text-right text-[11px] font-mono text-gray-400">
                    <div>{p.default_frame_count} frames</div>
                    <div className="text-indigo-400">{p.default_fps} FPS</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Animation Parameter Controls */}
          <div className="bg-[#121620] p-4 rounded-lg border border-[#252b38] flex flex-col gap-3 mt-1">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] font-medium text-gray-400 block mb-1">
                  Frame Count: <span className="text-indigo-400 font-bold">{frameCount}</span>
                </label>
                <input
                  type="range"
                  min="2"
                  max="16"
                  step="1"
                  value={frameCount}
                  onChange={(e) => setFrameCount(parseInt(e.target.value, 10))}
                  className="w-full accent-indigo-500"
                />
              </div>

              <div>
                <label className="text-[11px] font-medium text-gray-400 block mb-1">
                  Playback Speed: <span className="text-indigo-400 font-bold">{fps} FPS</span>
                </label>
                <input
                  type="range"
                  min="4"
                  max="24"
                  step="1"
                  value={fps}
                  onChange={(e) => setFps(parseInt(e.target.value, 10))}
                  className="w-full accent-indigo-500"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-gray-800">
              <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={isLooping}
                  onChange={(e) => setIsLooping(e.target.checked)}
                  className="accent-indigo-500 rounded"
                />
                <span>Loop Animation Seamlessly</span>
              </label>

              <button
                className="btn btn-primary text-xs font-semibold px-4 py-2 flex items-center gap-1.5"
                onClick={handleGenerateAnimation}
                disabled={isGenerating || !imageUrl}
              >
                {isGenerating ? (
                  <>
                    <span className="animate-spin">⏳</span>
                    <span>Rendering Frames...</span>
                  </>
                ) : (
                  <>
                    <span>🎬</span>
                    <span>Generate Animation</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Player & Active Frame Canvas */}
        <div className="col-span-7 flex flex-col gap-4">
          {/* Main Playback Viewport */}
          <div className="bg-[#0e1118] p-5 rounded-xl border border-[#202636] flex flex-col items-center justify-center">
            <div className="flex items-center justify-between w-full max-w-[280px] mb-2 text-xs">
              <span className="font-semibold text-gray-300 flex items-center gap-1.5">
                <span>Viewport</span>
                {isPlaying && (
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                )}
              </span>
              <span className="font-mono text-indigo-400">
                {frames.length > 0 ? `Frame ${currentFrameIdx + 1} / ${frames.length}` : 'No frames'}
              </span>
            </div>

            {/* Checkered Canvas Display */}
            <div className="pose-canvas-wrapper shadow-2xl relative">
              {activeFrame ? (
                <img
                  src={activeFrame.image_b64}
                  alt={`Frame ${currentFrameIdx + 1}`}
                  className="pose-sprite-img"
                  style={{ imageRendering: 'pixelated' }}
                />
              ) : imageUrl ? (
                <div className="relative w-full h-full flex flex-col items-center justify-center">
                  <img
                    src={imageUrl}
                    alt="Source Sprite Preview"
                    className="pose-sprite-img opacity-60"
                  />
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <span className="text-xs text-gray-300 font-medium bg-[#121620]/90 px-3 py-1.5 rounded border border-gray-700">
                      Click "Generate Animation" to render
                    </span>
                  </div>
                </div>
              ) : (
                <span className="text-xs text-gray-500">No Sprite Loaded</span>
              )}
            </div>

            {/* VCR Playback Controls */}
            <div className="flex items-center gap-2 mt-4">
              <button
                className="btn btn-secondary text-xs px-2.5 py-1.5"
                title="Reset to Frame 0"
                onClick={() => { setCurrentFrameIdx(0); }}
                disabled={frames.length === 0}
              >
                ↺
              </button>
              <button
                className="btn btn-secondary text-xs px-2.5 py-1.5"
                title="Previous Frame"
                onClick={() => setCurrentFrameIdx(prev => Math.max(0, prev - 1))}
                disabled={frames.length === 0 || currentFrameIdx === 0}
              >
                ⏮
              </button>
              <button
                className={`btn text-xs px-4 py-1.5 font-bold ${
                  isPlaying ? 'btn-secondary text-amber-300' : 'btn-primary'
                }`}
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={frames.length === 0}
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
              <button
                className="btn btn-secondary text-xs px-2.5 py-1.5"
                title="Next Frame"
                onClick={() => setCurrentFrameIdx(prev => Math.min(frames.length - 1, prev + 1))}
                disabled={frames.length === 0 || currentFrameIdx === frames.length - 1}
              >
                ⏭
              </button>

              <span className="text-xs font-mono text-gray-400 ml-2">
                {activeFrame ? `${activeFrame.timestamp_ms}ms` : '0ms'}
              </span>
            </div>

            {/* Scrubber Timeline Bar */}
            {frames.length > 0 && (
              <div className="w-full max-w-[320px] mt-3">
                <input
                  type="range"
                  min="0"
                  max={frames.length - 1}
                  value={currentFrameIdx}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setCurrentFrameIdx(parseInt(e.target.value, 10));
                  }}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>
            )}
          </div>

          {/* Quality Metrics Pill */}
          {animationResult && animationResult.overall_validation && (
            <div className="bg-[#121620] p-3 rounded-lg border border-[#252b38] flex items-center justify-between text-xs">
              <div className="flex items-center gap-4">
                <div>
                  <span className="text-gray-400 block text-[10px] uppercase">Avg IoU</span>
                  <span className="font-mono font-bold text-emerald-400">
                    {animationResult.overall_validation.silhouette_overlap_iou}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400 block text-[10px] uppercase">Min Alpha</span>
                  <span className="font-mono font-bold text-emerald-400">
                    {animationResult.overall_validation.alpha_integrity_pct}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-400 block text-[10px] uppercase">Total Time</span>
                  <span className="font-mono font-bold text-indigo-400">
                    {animationResult.processing_time_ms}ms
                  </span>
                </div>
              </div>

              <div className="text-[11px] text-gray-400">
                {animationResult.overall_validation.warnings.length === 0 ? (
                  <span className="text-emerald-400">✓ 0 distortion warnings</span>
                ) : (
                  <span className="text-amber-400">
                    ⚠️ {animationResult.overall_validation.warnings.length} warning(s)
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Frame Filmstrip Gallery */}
      {frames.length > 0 && (
        <div className="bg-[#121620] p-4 rounded-xl border border-[#252b38] flex flex-col gap-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs uppercase font-bold text-gray-400 tracking-wider">
              Frame Filmstrip ({frames.length} frames)
            </span>
            <span className="text-[11px] text-gray-500">
              Click frame thumbnail to inspect
            </span>
          </div>

          <div className="flex gap-2.5 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-700">
            {frames.map((frame, idx) => {
              const isActive = idx === currentFrameIdx;
              return (
                <div
                  key={frame.frame_index}
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentFrameIdx(idx);
                  }}
                  className={`flex flex-col items-center p-1.5 rounded-lg border cursor-pointer transition-all flex-shrink-0 ${
                    isActive
                      ? 'border-indigo-500 bg-indigo-950/40 shadow-[0_0_12px_rgba(99,102,241,0.3)]'
                      : 'border-gray-800 bg-[#0e1118] hover:border-gray-600'
                  }`}
                  style={{ width: '84px' }}
                >
                  <div
                    className="w-[72px] h-[72px] rounded overflow-hidden flex items-center justify-center"
                    style={{
                      backgroundColor: '#121620',
                      backgroundImage: 'linear-gradient(45deg, #181d28 25%, transparent 25%), linear-gradient(-45deg, #181d28 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #181d28 75%), linear-gradient(-45deg, transparent 75%, #181d28 75%)',
                      backgroundSize: '12px 12px'
                    }}
                  >
                    <img
                      src={frame.image_b64}
                      alt={`Frame ${idx + 1}`}
                      className="w-full h-full object-contain"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  </div>
                  <span className={`text-[10px] font-mono mt-1 ${isActive ? 'text-indigo-400 font-bold' : 'text-gray-400'}`}>
                    #{idx + 1} ({Math.round(frame.timestamp_ms)}ms)
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Spritesheet Export Drawer / Section */}
      {frames.length > 0 && (
        <div className="bg-[#121620] p-4 rounded-xl border border-[#252b38] flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-bold text-white flex items-center gap-1.5">
                <span>📦</span>
                <span>Spritesheet Packaging & Export</span>
              </h4>
              <p className="text-xs text-gray-400 mt-0.5">
                Pack all {frames.length} transparent frames into a production-ready game engine spritesheet PNG grid.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs">
                <span className="text-gray-400">Columns:</span>
                <select
                  value={sheetColumns}
                  onChange={(e) => setSheetColumns(parseInt(e.target.value, 10))}
                  className="bg-[#181d28] border border-gray-700 text-white text-xs rounded px-2 py-1"
                >
                  <option value="2">2 Columns</option>
                  <option value="4">4 Columns</option>
                  <option value="6">6 Columns</option>
                  <option value="8">8 Columns</option>
                </select>
              </div>

              <button
                className="btn btn-primary text-xs font-semibold px-4 py-1.5"
                onClick={handleExportSpritesheet}
                disabled={isExportingSheet}
              >
                {isExportingSheet ? 'Assembling Grid...' : 'Assemble Spritesheet'}
              </button>
            </div>
          </div>

          {/* Render Spritesheet Result if assembled */}
          {spritesheetResult && (
            <div className="p-4 bg-[#0e1118] border border-indigo-500/30 rounded-lg flex flex-col md:flex-row gap-6 items-center justify-between">
              {/* Preview Canvas */}
              <div
                className="rounded border border-gray-700 p-2 overflow-auto max-w-[320px] max-h-[180px]"
                style={{
                  backgroundColor: '#121620',
                  backgroundImage: 'linear-gradient(45deg, #181d28 25%, transparent 25%), linear-gradient(-45deg, #181d28 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #181d28 75%), linear-gradient(-45deg, transparent 75%, #181d28 75%)',
                  backgroundSize: '12px 12px'
                }}
              >
                <img
                  src={spritesheetResult.spritesheet_image_b64}
                  alt="Assembled Spritesheet"
                  style={{ imageRendering: 'pixelated' }}
                  className="max-w-none"
                />
              </div>

              {/* Metadata Badges */}
              <div className="flex flex-col gap-2 flex-grow text-xs text-gray-300">
                <div className="flex gap-4">
                  <div>
                    <span className="text-[10px] uppercase text-gray-500 block">Total Dimensions</span>
                    <span className="font-mono font-bold text-white">
                      {spritesheetResult.metadata.total_width} × {spritesheetResult.metadata.total_height}px
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-gray-500 block">Grid Layout</span>
                    <span className="font-mono font-bold text-indigo-400">
                      {spritesheetResult.metadata.columns} cols × {spritesheetResult.metadata.rows} rows
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-gray-500 block">Frame Size</span>
                    <span className="font-mono font-bold text-white">
                      {spritesheetResult.metadata.frame_width} × {spritesheetResult.metadata.frame_height}px
                    </span>
                  </div>
                </div>
                <p className="text-[11px] text-gray-400">
                  {spritesheetResult.message}
                </p>
              </div>

              {/* Download Action */}
              <button
                className="btn btn-primary text-xs font-bold px-4 py-2 flex items-center gap-1.5 whitespace-nowrap"
                onClick={handleDownloadSheet}
              >
                <span>💾</span>
                <span>Download PNG Spritesheet</span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Error notification banner */}
      {error && (
        <div className="p-3 bg-red-950/40 border border-red-500/40 rounded-lg text-xs text-red-300 flex items-center gap-2">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};

export default AnimationTimeline;
