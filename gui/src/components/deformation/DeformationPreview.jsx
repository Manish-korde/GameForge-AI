import React from 'react';

/**
 * DeformationPreview component: renders deformed sprite preview and QA distortion audit card.
 */
const DeformationPreview = ({
  imageUrl,
  result,
  viewMode,
  assetName,
  onExportPng
}) => {
  return (
    <div className="flex flex-col gap-4">
      {/* Right Canvas: Deformed Result or Placeholder */}
      <div className="flex flex-col items-center">
        <span className="text-[11px] uppercase font-semibold text-gray-400 mb-1.5 flex items-center gap-1.5">
          <span>Deformed Output</span>
          {result && (
            <span className="text-[10px] text-emerald-400 font-mono">
              ({result.processing_time_ms}ms)
            </span>
          )}
        </span>

        <div className="pose-canvas-wrapper relative" style={{ width: '256px', height: '256px' }}>
          {result && result.output_image_b64 ? (
            <>
              {/* Ghost overlay mode */}
              {viewMode === 'ghost_overlay' && (
                <img
                  src={imageUrl}
                  alt="Ghost Source"
                  className="pose-sprite-img absolute inset-0 opacity-30"
                />
              )}
              <img
                src={result.output_image_b64}
                alt="Deformed Result"
                className="pose-sprite-img relative z-10"
              />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center p-4 text-center text-gray-500">
              <span className="text-3xl mb-2 opacity-40">🧙‍♂️</span>
              <p className="text-xs">Click "Run Deformation" to synthesize target pose</p>
            </div>
          )}
        </div>
      </div>

      {/* Validation & Quality Assurance Audit Panel */}
      {result && (
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Quality & Distortion Audit
              </span>
              {result.validation?.is_valid ? (
                <span className="badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px] px-2 py-0.5 rounded">
                  Passed Verification
                </span>
              ) : (
                <span className="badge bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[10px] px-2 py-0.5 rounded">
                  Distortion Flagged
                </span>
              )}
            </div>

            <button
              className="btn btn-primary text-xs px-3 py-1.5 flex items-center gap-1.5"
              onClick={onExportPng}
            >
              <span>💾</span>
              <span>Export Transparent PNG</span>
            </button>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-4 gap-2 mb-3">
            <div className="metric-pill">
              <span className="metric-pill-label">Silhouette IoU</span>
              <span className="metric-pill-value text-indigo-400">
                {result.validation?.silhouette_overlap_iou?.toFixed(3) || 'N/A'}
              </span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill-label">Alpha Integrity</span>
              <span className="metric-pill-value text-emerald-400">
                {result.validation?.alpha_integrity_pct?.toFixed(1) || '100.0'}%
              </span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill-label">Max Bone Stretch</span>
              <span className="metric-pill-value text-purple-400">
                {result.validation?.max_bone_stretch_ratio?.toFixed(2) || '1.00'}x
              </span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill-label">Canvas Clipping</span>
              <span className={`metric-pill-value ${result.validation?.clipping_detected ? 'text-amber-400' : 'text-emerald-400'}`}>
                {result.validation?.clipping_detected ? 'Detected' : 'None (Safe)'}
              </span>
            </div>
          </div>

          {/* Diagnostic Warnings */}
          {result.validation?.warnings && result.validation.warnings.length > 0 && (
            <div className="bg-[#181d28] border border-[#2b3345] rounded p-2.5 text-[11px] text-gray-300">
              <div className="font-semibold text-amber-400 mb-1">Diagnostic Warnings:</div>
              <ul className="list-disc list-inside space-y-0.5">
                {result.validation.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DeformationPreview;
