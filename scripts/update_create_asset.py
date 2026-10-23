with open('gui/src/pages/CreateAsset.jsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add import
if 'import PoseDeformationStudio' not in code:
    code = code.replace(
        "import { useState, useEffect } from 'react';",
        "import { useState, useEffect } from 'react';\nimport PoseDeformationStudio from '../components/PoseDeformationStudio';"
    )

# 2. Add Sidebar Button
target_btn = """<button className="btn btn-secondary w-full mb-2" onClick={() => setActiveView('vae')} disabled={isAELoading || isVAELoading}>
                  Create VAE Variations
                </button>"""

replacement_btn = """<button className="btn btn-secondary w-full mb-2" onClick={() => setActiveView('vae')} disabled={isAELoading || isVAELoading}>
                  Create VAE Variations
                </button>
                <button 
                  className={`btn ${activeView === 'deformation' ? 'btn-primary' : 'btn-secondary'} w-full mb-2 flex items-center justify-center gap-2`}
                  onClick={() => setActiveView('deformation')} 
                  disabled={isAELoading || isVAELoading}
                >
                  <span>🦴</span>
                  <span>Pose Deformation</span>
                </button>"""

# Normalize \r\n to \n for replacement
code_normalized = code.replace('\r\n', '\n')
if '<span>Pose Deformation</span>' not in code_normalized:
    code_normalized = code_normalized.replace(target_btn, replacement_btn)

# 3. Add Studio view when activeView === 'deformation'
target_vae_end = """                  <div className="flex gap-md justify-center mt-8">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setActiveView('original')}>Back to Original</button>
                  </div>
                </div>
              )}"""

replacement_vae_end = """                  <div className="flex gap-md justify-center mt-8">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setActiveView('original')}>Back to Original</button>
                  </div>
                </div>
              )}

              {activeView === 'deformation' && (
                <PoseDeformationStudio
                  imageUrl={generatedAsset.imageUrl}
                  assetName={generatedAsset.name}
                  onBack={() => setActiveView('original')}
                />
              )}"""

if "{activeView === 'deformation' && (" not in code_normalized:
    code_normalized = code_normalized.replace(target_vae_end, replacement_vae_end)

with open('gui/src/pages/CreateAsset.jsx', 'w', encoding='utf-8') as f:
    f.write(code_normalized)

print("CreateAsset.jsx updated successfully!")
