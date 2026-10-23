import React from 'react';

/**
 * DeformationControls component: configuration sliders, method toggles, and execution actions.
 */
const DeformationControls = ({
  intensity,
  onChangeIntensity,
  method,
  onChangeMethod,
  resampling,
  onChangeResampling,
  preservePixelArt,
  onChangePreservePixelArt,
  isLoading,
  isDetecting,
  onRunDeformation,
  onResetPose
}) => {
  return (
    <div className="card p-4 flex flex-col gap-3">
      <label className="text-xs font-semibold text-white uppercase tracking-wider">
        Transformation Parameters
      </label>

      {/* Intensity Slider */}
      <div>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-gray-300">Pose Intensity (Weight)</span>
          <span className="font-mono text-indigo-400 font-semibold">{intensity.toFixed(2)}x</span>
        </div>
        <input
          type="range"
          min="0.0"
          max="1.5"
          step="0.05"
          value={intensity}
          onChange={e => onChangeIntensity(parseFloat(e.target.value))}
          className="w-full accent-indigo-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
          <span>0.0 (Identity)</span>
          <span>1.0 (Full Target)</span>
          <span>1.5 (Exaggerated)</span>
        </div>
      </div>

      {/* Algorithm Method Toggle */}
      <div>
        <label className="block text-xs text-gray-300 mb-1">Deformation Algorithm</label>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            className={`text-xs py-2 px-2.5 rounded border text-left transition-all ${
              method === 'geometric'
                ? 'border-indigo-500 bg-indigo-500/15 text-white font-semibold shadow-sm'
                : 'border-[#2b3345] bg-[#121620] text-gray-400 hover:text-white'
            }`}
            onClick={() => onChangeMethod('geometric')}
          >
            <div className="flex items-center gap-1.5">
              <span className="text-emerald-400">●</span>
              <span>Geometric Skinning</span>
            </div>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Dependable & crisp pixel-art
            </span>
          </button>

          <button
            type="button"
            className={`text-xs py-2 px-2.5 rounded border text-left transition-all ${
              method === 'hybrid'
                ? 'border-purple-500 bg-purple-500/15 text-white font-semibold shadow-sm'
                : 'border-[#2b3345] bg-[#121620] text-gray-400 hover:text-white'
            }`}
            onClick={() => onChangeMethod('hybrid')}
          >
            <div className="flex items-center gap-1.5">
              <span className="text-purple-400">●</span>
              <span>Hybrid Geometric+VAE</span>
            </div>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Experimental 280K VAE
            </span>
          </button>
        </div>
      </div>

      {/* Resampling & Options */}
      <div className="grid grid-cols-2 gap-2 pt-1">
        <div>
          <label className="block text-[11px] text-gray-400 mb-1">Resampling Policy</label>
          <select
            value={resampling}
            onChange={e => onChangeResampling(e.target.value)}
            className="bg-[#121620] border border-[#2b3345] text-xs text-white rounded p-1.5 w-full focus:outline-none focus:border-indigo-500"
          >
            <option value="nearest">Nearest (Pixel Art)</option>
            <option value="bilinear">Bilinear (Smooth)</option>
          </select>
        </div>

        <div className="flex items-center gap-2 pt-4">
          <input
            type="checkbox"
            id="pixelArtCheck"
            checked={preservePixelArt}
            onChange={e => onChangePreservePixelArt(e.target.checked)}
            className="accent-indigo-500 rounded"
          />
          <label htmlFor="pixelArtCheck" className="text-xs text-gray-300 cursor-pointer select-none">
            Preserve Alpha Edges
          </label>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2 mt-2 pt-2 border-t border-[#252b38]">
        <button
          className="btn btn-primary flex-1 py-2 text-xs font-semibold flex items-center justify-center gap-2"
          onClick={onRunDeformation}
          disabled={isLoading || isDetecting}
        >
          {isLoading ? (
            <>
              <span className="animate-spin text-sm">⏳</span>
              <span>Deforming Sprite...</span>
            </>
          ) : (
            <>
              <span>⚡ Run Deformation</span>
            </>
          )}
        </button>

        <button
          className="btn btn-secondary px-3 py-2 text-xs"
          onClick={onResetPose}
          disabled={isLoading}
          title="Reset target pose"
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export default DeformationControls;
