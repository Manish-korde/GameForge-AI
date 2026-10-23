import React from 'react';

/**
 * PoseSelector component: displays catalog of pre-tested target pose templates.
 */
const PoseSelector = ({ templates, selectedTemplate, onSelectTemplate }) => {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <label className="text-xs font-semibold text-white uppercase tracking-wider">
          Select Target Pose
        </label>
        <span className="text-[10px] text-secondary font-mono">
          {templates.length} Templates Tested
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 max-h-[220px] overflow-y-auto pr-1">
        {templates.map(tpl => (
          <div
            key={tpl.id}
            className={`template-card ${selectedTemplate === tpl.id ? 'active' : ''}`}
            onClick={() => onSelectTemplate(tpl.id)}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white">{tpl.name}</span>
              <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-[#1e2330] text-gray-300">
                {tpl.category}
              </span>
            </div>
            <p className="text-[10px] text-gray-400 mt-1 line-clamp-1">{tpl.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PoseSelector;
