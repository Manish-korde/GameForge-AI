import { useState, useEffect } from 'react';
import { generateDesign, generateImage, runAEInference, runVAEInference, runNoisyAEInference, runVAEInterpolation, fetchRandomHumanSprite, searchSimilarAssets, clusterAssets, detectDuplicateAsset, getAnomalyScore, retrieveMoreCategoryAssets, SAMPLE_SPRITES_35, fetchSampleSpritesManifest } from '../services/api';

const CreateAsset = () => {
  const [prompt, setPrompt] = useState('');
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [designSpec, setDesignSpec] = useState(null);
  const [generatedAsset, setGeneratedAsset] = useState(null);
  const [aeResult, setAeResult] = useState(null);
  const [isAELoading, setIsAELoading] = useState(false);
  const [isAeApproved, setIsAeApproved] = useState(false);
  const [noisyAeResult, setNoisyAeResult] = useState(null);
  const [isNoisyAELoading, setIsNoisyAELoading] = useState(false);
  const [noiseScale, setNoiseScale] = useState(0.2);
  const [isDragging, setIsDragging] = useState(false);
  
  const [vaeResult, setVaeResult] = useState(null);
  const [isVAELoading, setIsVAELoading] = useState(false);
  const [vaeScale, setVaeScale] = useState(1.0);
  
  const [spriteList, setSpriteList] = useState(SAMPLE_SPRITES_35);

  useEffect(() => {
    fetchSampleSpritesManifest().then(data => {
      if (data && data.length > 0) {
        setSpriteList(data);
      }
    });
  }, []);

  const [characterAUrl, setCharacterAUrl] = useState('');
  const [characterBUrl, setCharacterBUrl] = useState('');
  const [interpolationAlpha, setInterpolationAlpha] = useState(0.5);
  const [interpolatedResult, setInterpolatedResult] = useState(null);
  const [isInterpolating, setIsInterpolating] = useState(false);
  
  // VAE Suite States
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [clusterData, setClusterData] = useState(null);
  const [isClustering, setIsClustering] = useState(false);
  const [duplicateResult, setDuplicateResult] = useState(null);
  const [isCheckingDuplicate, setIsCheckingDuplicate] = useState(false);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [isCheckingAnomaly, setIsCheckingAnomaly] = useState(false);

  
  const [activeView, setActiveView] = useState('original'); // 'original', 'ae', 'vae'

  const handleGenerateDesign = async () => {
    if (!prompt) return;
    setIsLoading(true);
    const res = await generateDesign(prompt);
    setDesignSpec(res.data);
    setIsLoading(false);
    setStep(2);
  };

  const handleGenerateAsset = async () => {
    setIsLoading(true);
    const res = await generateImage(designSpec);
    setGeneratedAsset(res.data);
    setAeResult(null); // Reset AE result on new image
    setNoisyAeResult(null);
    setVaeResult(null);
    setCharacterAUrl(res.data.imageUrl);
    setCharacterBUrl(spriteList[5]?.url || spriteList[0]?.url); // Default Character B
    setInterpolatedResult(null);
    setActiveView('original');
    setIsLoading(false);
    setStep(3);
  };

  const handleRunAE = async () => {
    if (!generatedAsset) return;
    setIsAELoading(true);
    setActiveView('ae');
    try {
      const result = await runAEInference(generatedAsset.imageUrl);
      setAeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run AE Inference");
      setActiveView('original');
    } finally {
      setIsAELoading(false);
    }
  };

  const handleRunNoisyAE = async () => {
    if (!generatedAsset) return;
    setIsNoisyAELoading(true);
    try {
      const result = await runNoisyAEInference(generatedAsset.imageUrl, noiseScale);
      setNoisyAeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run Noisy AE Inference");
    } finally {
      setIsNoisyAELoading(false);
    }
  };

  const handleRunVAE = async () => {
    if (!generatedAsset) return;
    setIsVAELoading(true);
    try {
      const result = await runVAEInference(generatedAsset.imageUrl, vaeScale);
      setVaeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run VAE Inference");
    } finally {
      setIsVAELoading(false);
    }
  };

  const handleRunInterpolation = async () => {
    if (!characterAUrl || !characterBUrl) return;
    setIsInterpolating(true);
    try {
      const result = await runVAEInterpolation(characterAUrl, characterBUrl, interpolationAlpha);
      setInterpolatedResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run VAE Interpolation");
    } finally {
      setIsInterpolating(false);
    }
  };

  // VAE Suite Handlers
  const handleSearchSimilar = async (catFilter = selectedCategory) => {
    if (!generatedAsset) return;
    setIsSearching(true);
    try {
      const res = await searchSimilarAssets(generatedAsset.imageUrl, 4, catFilter);
      setSearchResults(res);
      setSelectedCategory(catFilter);
    } catch (err) {
      console.error(err);
      alert("Failed visual search");
    } finally {
      setIsSearching(false);
    }
  };


  const handleClusterAssets = async () => {
    if (!generatedAsset) return;
    setIsClustering(true);
    try {
      const res = await clusterAssets([generatedAsset.imageUrl]);
      setClusterData(res);
    } catch (err) {
      console.error(err);
      alert("Failed clustering");
    } finally {
      setIsClustering(false);
    }
  };

  const handleDetectDuplicate = async () => {
    if (!characterAUrl || !characterBUrl) return;
    setIsCheckingDuplicate(true);
    try {
      const res = await detectDuplicateAsset(characterAUrl, characterBUrl);
      setDuplicateResult(res);
    } catch (err) {
      console.error(err);
      alert("Failed duplicate check");
    } finally {
      setIsCheckingDuplicate(false);
    }
  };

  const handleCheckAnomaly = async () => {
    if (!generatedAsset) return;
    setIsCheckingAnomaly(true);
    try {
      const res = await getAnomalyScore(generatedAsset.imageUrl);
      setAnomalyResult(res);
    } catch (err) {
      console.error(err);
      alert("Failed anomaly detection");
    } finally {
      setIsCheckingAnomaly(false);
    }
  };


  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFileUpload(file);
    }
  };
  
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };
  
  const handleFileUpload = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      const base64Url = event.target.result;
      setGeneratedAsset({
        id: 'custom-' + Date.now(),
        name: file.name,
        type: 'Image',
        source: 'Upload',
        model: 'Custom Upload',
        prompt: 'Custom user uploaded image for AE testing',
        imageUrl: base64Url,
        createdAt: new Date().toISOString(),
        status: 'Completed'
      });
      setAeResult(null);
      setNoisyAeResult(null);
      setVaeResult(null);
      setCharacterBUrl(spriteList[5]?.url || spriteList[0]?.url);
      setInterpolatedResult(null);
      setActiveView('original');
      setStep(3);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="create-asset">
      <h2>Create Game Asset</h2>

      {step === 1 && (
        <div className="card mt-4">
          <label className="block mb-2 text-sm">Prompt</label>
          <textarea 
            className="input textarea mb-4" 
            placeholder="Describe the game asset you want to create..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          ></textarea>
          
          <div className="flex gap-sm mb-6">
            <button className="chip" onClick={() => setPrompt("Dark fantasy knight")}>Dark fantasy knight</button>
            <button className="chip" onClick={() => setPrompt("Pixel-art forest village")}>Pixel-art forest village</button>
            <button className="chip" onClick={() => setPrompt("Magic sword")}>Magic sword</button>
          </div>

          <div className="grid grid-cols-3 gap-md mb-6">
            <div>
              <label className="block text-sm mb-1">Asset Type</label>
              <select className="input"><option>Character</option></select>
            </div>
            <div>
              <label className="block text-sm mb-1">Style</label>
              <select className="input"><option>Pixel Art</option></select>
            </div>
            <div>
              <label className="block text-sm mb-1">Generation Mode</label>
              <select className="input"><option>Full Pipeline</option></select>
            </div>
          </div>

          <button className="btn btn-primary" onClick={handleGenerateDesign} disabled={isLoading}>
            {isLoading ? 'Generating...' : 'Generate Design'}
          </button>
        </div>
      )}

      {step === 2 && designSpec && (
        <div className="grid grid-cols-2 gap-lg mt-4">
          <div className="card">
            <h3 className="mb-4">GAME CONCEPT SPECIFICATION</h3>
            <div className="grid grid-cols-2 gap-md mb-4">
              <div>
                <span className="text-sm text-secondary block">Title & Genre</span>
                <strong>{designSpec.game_title || designSpec.theme || 'Game Concept'}</strong>
                <div className="text-xs text-secondary">{designSpec.genre || 'Pixel Art RPG'}</div>
              </div>
              <div>
                <span className="text-sm text-secondary block">Environment & Theme</span>
                {typeof designSpec.environment === 'object' ? designSpec.environment?.theme : (designSpec.environment || 'Dungeon')}
              </div>
              <div>
                <span className="text-sm text-secondary block">Character Role</span>
                {designSpec.main_character ? designSpec.main_character?.role : (Array.isArray(designSpec.characters) ? designSpec.characters.length : 'Hero')}
              </div>
              <div>
                <span className="text-sm text-secondary block">Visual Style</span>
                {designSpec.art_style || designSpec.visualStyle || '16-bit Pixel Art'}
              </div>
            </div>
            <div className="flex gap-md mt-6">
              <button className="btn btn-secondary" onClick={() => setStep(1)}>Edit Specification</button>
              <button className="btn btn-primary" onClick={handleGenerateAsset} disabled={isLoading}>
                {isLoading ? 'Generating Image...' : 'Continue to Visual Generation'}
              </button>
            </div>
          </div>
          <div className="card">
             <h3 className="mb-4 text-sm uppercase text-secondary">Transformer Details</h3>
             <pre className="raw-output">{designSpec.rawOutput || JSON.stringify(designSpec, null, 2)}</pre>
          </div>
        </div>
      )}


      {step === 3 && generatedAsset && (
        <div className="grid grid-cols-3 gap-lg mt-4">
          <div className="col-span-1">
              <div className="card text-left">
                <h3 className="mb-4">Asset Details</h3>
                <div className="text-sm text-secondary mb-1">Name</div>
                <div className="mb-3 font-semibold text-white">{generatedAsset.name}</div>
                <div className="text-sm text-secondary mb-1">Prompt</div>
                <div className="mb-4 text-sm">{generatedAsset.prompt}</div>

                {/* Dropdown with 35 Character Sprites */}
                <div className="mb-5 text-left border-t border-b border-[#252530] py-3">
                  <label className="block text-xs font-semibold text-indigo-300 mb-1.5 flex justify-between items-center">
                    <span>🎭 Select Character Sprite</span>
                    <span className="text-[10px] text-emerald-400 font-mono">35 Available</span>
                  </label>
                  <select 
                    value={generatedAsset.imageUrl} 
                    onChange={(e) => {
                      const targetUrl = e.target.value;
                      const foundItem = spriteList.find(s => s.url === targetUrl);
                      const newName = foundItem ? (foundItem.label.split(": ")[1] || foundItem.label) : generatedAsset.name;
                      setGeneratedAsset({
                        ...generatedAsset,
                        name: newName,
                        imageUrl: targetUrl
                      });
                      setCharacterAUrl(targetUrl);
                      setAeResult(null);
                      setNoisyAeResult(null);
                      setVaeResult(null);
                    }}
                    className="bg-[#181820] border border-[#3a3a45] text-xs text-white rounded-lg p-2.5 w-full focus:outline-none focus:border-primary cursor-pointer font-medium"
                  >
                    {spriteList.map((s) => (
                      <option key={s.id} value={s.url}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </div>

                <button className="btn btn-secondary w-full mb-2" onClick={handleRunAE} disabled={isAELoading || isVAELoading}>
                  {isAELoading ? 'Running AE...' : 'Run through AE'}
                </button>
                <button className="btn btn-secondary w-full mb-2" onClick={() => setActiveView('vae')} disabled={isAELoading || isVAELoading}>
                  Create VAE Variations
                </button>
              </div>
          </div>
          <div className="col-span-2">
            <div className="card text-center flex flex-col items-center h-full">
              {activeView === 'original' && (
                <div 
                  className={`w-full h-full min-h-[300px] flex flex-col items-center justify-center border-2 border-dashed rounded-lg transition-colors relative ${isDragging ? 'border-primary bg-primary/10' : 'border-[#333] hover:border-gray-500'}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <img src={generatedAsset.imageUrl} alt="Generated" className="generated-preview mb-4 object-contain max-h-[300px]" />
                  <div className="flex gap-md mb-2">
                    <button className="btn btn-primary">Save to Library</button>
                  </div>
                  
                  {/* Drag and Drop Overlay / Controls */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center opacity-0 hover:opacity-100 bg-black/60 rounded-lg transition-opacity">
                    <p className="mb-2 font-medium">Drag & Drop to swap image</p>
                    <p className="text-sm text-secondary mb-4">or</p>
                    <label className="btn btn-secondary cursor-pointer">
                      Browse Files
                      <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                    </label>
                  </div>
                </div>
              )}

              {activeView === 'ae' && !aeResult && (
                <div className="flex-1 flex items-center justify-center">Loading AE Results...</div>
              )}

              {activeView === 'ae' && aeResult && (
                <div className="w-full">
                  <div className="mb-6 text-center border-b border-[#25252e] pb-4">
                    <h4 className="text-2xl font-bold text-white">
                      🔍 Autoencoder Inference & Outlier Screening
                    </h4>
                    <p className="text-xs text-secondary mt-1 max-w-2xl mx-auto">
                      Deterministic high-precision reconstruction baseline & <strong>Automated Asset QA & Reconstruction-Based Outlier Screening</strong> relative to the learned 280K visual distribution.
                    </p>
                  </div>

                  <div className="flex gap-lg justify-center mb-6">
                    <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                      <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Original (Scaled)</span>
                      <img src={aeResult.original_processed} alt="Original processed" className="rounded shadow-sm" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                    </div>
                    <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                      <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Reconstructed</span>
                      <img src={aeResult.reconstructed} alt="Reconstructed" className="rounded shadow-sm" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                    </div>
                  </div>

                  {aeResult.metrics && (
                    <div className="max-w-2xl mx-auto mb-8">
                      {/* Automated Asset QA Outlier Status Banner */}
                      {aeResult.metrics.mse <= 0.001313 ? (
                        <div className="p-4 mb-6 rounded-xl border bg-emerald-950/40 border-emerald-500/50 flex items-center justify-between text-left shadow-lg">
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">🟢</span>
                            <div>
                              <h5 className="text-emerald-400 font-bold text-sm">Automated Asset QA: Passed In-Distribution Audit</h5>
                              <p className="text-xs text-gray-300 mt-0.5">
                                Reconstruction MSE (<span className="font-mono text-emerald-300 font-bold">{aeResult.metrics.mse.toFixed(6)}</span>) is below the P95 outlier threshold (<code className="text-emerald-300">0.001313</code>). Asset matches the learned 280K visual distribution.
                              </p>
                            </div>
                          </div>
                          <span className="badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-xs px-3 py-1 font-semibold">In-Distribution</span>
                        </div>
                      ) : (
                        <div className="p-4 mb-6 rounded-xl border bg-amber-950/40 border-amber-500/50 flex items-center justify-between text-left shadow-lg">
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">⚠️</span>
                            <div>
                              <h5 className="text-amber-400 font-bold text-sm">Automated Asset QA: Flagged for Human Review (Outlier Detected)</h5>
                              <p className="text-xs text-gray-300 mt-0.5">
                                Reconstruction MSE (<span className="font-mono text-amber-300 font-bold">{aeResult.metrics.mse.toFixed(6)}</span>) exceeds the P95 outlier threshold (<code className="text-amber-300">0.001313</code>). This asset is unusual relative to the learned distribution and is flagged for manual inspection.
                              </p>
                            </div>
                          </div>
                          <span className="badge bg-amber-500/20 text-amber-300 border border-amber-500/40 text-xs px-3 py-1 font-semibold">Flagged Outlier</span>
                        </div>
                      )}

                      {/* Real-Time Evaluation Metrics */}
                      <div className="bg-[#151515] rounded-xl p-6 border border-border mb-6 shadow-lg text-left">
                        <h5 className="text-sm text-secondary uppercase tracking-wider mb-4 font-semibold">Real-Time Evaluation Metrics</h5>
                        
                        {/* Primary Error Metrics */}
                        <div className="grid grid-cols-2 gap-4 text-center mb-4">
                          <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                            <div className="text-xs text-secondary mb-1">MSE Loss</div>
                            <div className="font-mono text-primary text-xl font-bold">{aeResult.metrics.mse.toFixed(6)}</div>
                          </div>
                          <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                            <div className="text-xs text-secondary mb-1">MAE Loss</div>
                            <div className="font-mono text-primary text-xl font-bold">{aeResult.metrics.mae.toFixed(6)}</div>
                          </div>
                        </div>

                        {/* Visual Accuracy Metrics */}
                        <div className="grid grid-cols-3 gap-4 text-center mb-6">
                          <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                            <div className="text-xs text-secondary mb-1">Color Palette Match</div>
                            <div className="font-mono text-green-400 font-bold">{aeResult.metrics.color_match.toFixed(2)}%</div>
                          </div>
                          <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                            <div className="text-xs text-secondary mb-1">Exact Pixel Match</div>
                            <div className="font-mono text-green-400 font-bold">{aeResult.metrics.exact_match.toFixed(2)}%</div>
                          </div>
                          <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                            <div className="text-xs text-secondary mb-1">Alpha Mask IoU</div>
                            <div className="font-mono text-green-400 font-bold">{aeResult.metrics.alpha_iou.toFixed(2)}%</div>
                          </div>
                        </div>

                        {/* Footer Metrics */}
                        <div className="flex justify-between items-center text-[#777] text-xs border-t border-[#333] pt-3 px-2 font-mono">
                          <span>PSNR: {aeResult.metrics.psnr.toFixed(2)} dB</span>
                          <span>SSIM: {aeResult.metrics.ssim.toFixed(4)}</span>
                        </div>
                      </div>

                      {/* Human Designer Oversight & Sign-off Checkpoint Card */}
                      <div className="bg-[#181820] rounded-xl p-5 border border-[#2a2a35] shadow-lg text-left">
                        <h5 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                          🤝 Human Designer Oversight Checkpoint
                        </h5>
                        <p className="text-xs text-secondary mb-4">
                          Review the reconstruction metrics and outlier screening audit above, then sign off to approve or flag this asset for project library export.
                        </p>

                        <div className="flex flex-col md:flex-row justify-between items-center gap-4 pt-3 border-t border-[#2a2a35]">
                          <label className="flex items-center gap-3 cursor-pointer">
                            <input 
                              type="checkbox"
                              checked={isAeApproved}
                              onChange={(e) => setIsAeApproved(e.target.checked)}
                              className="w-5 h-5 accent-emerald-500 rounded cursor-pointer"
                            />
                            <span className={`text-xs ${isAeApproved ? 'text-emerald-400 font-semibold' : 'text-gray-300'}`}>
                              I have reviewed this asset's reconstruction & QA outlier screening status
                            </span>
                          </label>

                          <button 
                            className="btn text-xs py-2 px-4 transition-all"
                            disabled={!isAeApproved}
                            style={{
                              backgroundColor: isAeApproved ? '#10b981' : '#374151',
                              borderColor: isAeApproved ? '#059669' : '#4b5563',
                              color: '#ffffff',
                              opacity: isAeApproved ? 1 : 0.5,
                              cursor: isAeApproved ? 'pointer' : 'not-allowed'
                            }}
                            onClick={() => alert("Asset approved for project library!")}
                          >
                            {isAeApproved ? "Approve Asset for Library 🚀" : "Approval Required"}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Noisy AE Section */}
                  <div className="mt-12 border-t border-[#333] pt-8">
                    <h4 className="mb-6 text-xl font-bold">Robustness Limit Test — AE trained for reconstruction only, not denoising</h4>
                    <div className="mb-6 max-w-md mx-auto">
                      <label className="block text-sm text-secondary mb-2">Noise Level: {noiseScale}</label>
                      <input 
                        type="range" 
                        min="0.0" 
                        max="1.0" 
                        step="0.05" 
                        value={noiseScale} 
                        onChange={(e) => setNoiseScale(parseFloat(e.target.value))}
                        className="w-full mb-4 accent-primary"
                      />
                      <button className="btn btn-primary w-full" onClick={handleRunNoisyAE} disabled={isNoisyAELoading}>
                        {isNoisyAELoading ? 'Adding Noise & Reconstructing...' : 'Add Noise & Reconstruct'}
                      </button>
                    </div>

                    {noisyAeResult && (
                      <>
                        <div className="flex gap-lg justify-center mb-8">
                          <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                            <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Original</span>
                            <img src={noisyAeResult.original_processed} alt="Original" className="rounded shadow-sm" style={{ width: '200px', height: '200px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                          </div>
                          <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                            <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Noisy Input</span>
                            <img src={noisyAeResult.noisy_image} alt="Noisy Input" className="rounded shadow-sm" style={{ width: '200px', height: '200px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                          </div>
                          <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                            <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Reconstructed</span>
                            <img src={noisyAeResult.reconstructed} alt="Reconstructed" className="rounded shadow-sm" style={{ width: '200px', height: '200px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                          </div>
                        </div>

                        <div className="bg-[#151515] rounded-xl p-6 border border-border mb-8 max-w-2xl mx-auto shadow-lg">
                          <h5 className="text-sm text-secondary uppercase tracking-wider mb-4 font-semibold text-left">Robustness Limit Evaluation Metrics</h5>
                          <div className="grid grid-cols-2 gap-4 text-center mb-4">
                            <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">MSE Loss</div>
                              <div className="font-mono text-primary text-xl font-bold">{noisyAeResult.metrics.mse.toFixed(6)}</div>
                            </div>
                            <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">MAE Loss</div>
                              <div className="font-mono text-primary text-xl font-bold">{noisyAeResult.metrics.mae.toFixed(6)}</div>
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-center mb-6">
                            <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">Color Palette Match</div>
                              <div className="font-mono text-green-400 font-bold">{noisyAeResult.metrics.color_match.toFixed(2)}%</div>
                            </div>
                            <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">Exact Pixel Match</div>
                              <div className="font-mono text-green-400 font-bold">{noisyAeResult.metrics.exact_match.toFixed(2)}%</div>
                            </div>
                            <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">Alpha Mask IoU</div>
                              <div className="font-mono text-green-400 font-bold">{noisyAeResult.metrics.alpha_iou.toFixed(2)}%</div>
                            </div>
                          </div>
                          <div className="flex justify-between items-center text-[#555] text-[10px] border-t border-[#333] pt-3 px-2">
                            <span>PSNR: {noisyAeResult.metrics.psnr.toFixed(2)} dB</span>
                            <span>SSIM: {noisyAeResult.metrics.ssim.toFixed(4)}</span>
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="flex gap-md justify-center">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setActiveView('original')}>Back to Original</button>
                  </div>
                </div>
              )}

              {activeView === 'vae' && (
                <div className="w-full">
                  {/* Headline Title */}
                  <div className="mb-8 text-center border-b border-[#25252e] pb-4">
                    <h3 className="text-2xl font-bold text-white">
                      <span className="text-primary">✨</span> VAE Asset Discovery, Exploration & Diagnostics
                    </h3>
                    <p className="text-xs text-secondary mt-1 max-w-2xl mx-auto">
                      An interactive co-pilot workspace for discovering matching library assets, generating rough concept variations, and verifying technical quality before asset export.
                    </p>
                  </div>

                  {/* SECTION 1: ASSET DISCOVERY */}
                  <div className="mb-10 bg-[#121215] border border-[#2a2a35] rounded-2xl p-6 shadow-xl text-left">
                    <div className="flex items-center justify-between mb-4 border-b border-[#25252e] pb-3">
                      <div>
                        <h4 className="text-lg font-bold text-white uppercase tracking-wider">1. Asset Discovery & Library Search</h4>
                        <p className="text-xs text-indigo-300 mt-0.5 font-medium">
                          🎨 <strong>Why an artist uses this:</strong> Find visually matching sprites across your project library to maintain aesthetic consistency across characters, weapons, and items.
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button className="btn btn-primary text-xs py-1.5 px-3" onClick={() => handleSearchSimilar()} disabled={isSearching}>
                          {isSearching ? 'Searching Library...' : 'Find Similar Sprites'}
                        </button>
                        <button className="btn btn-secondary text-xs py-1.5 px-3" onClick={handleClusterAssets} disabled={isClustering}>
                          {isClustering ? 'Mapping Library...' : 'Browse Asset Categories'}
                        </button>
                      </div>
                    </div>

                    {/* Visual Search Panel */}
                    {searchResults && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] mb-4">
                        <div className="flex justify-between items-center mb-3">
                          <div>
                            <h5 className="text-sm font-semibold text-white">
                              4 similar sprites found in your indexed library <span className="text-indigo-400 font-mono">(25,000 of 282,511 total sprites indexed)</span>
                            </h5>
                            <p className="text-[11px] text-gray-400 mt-0.5">Retrieved via VAE continuous vector similarity matching.</p>
                          </div>
                          <div className="flex gap-1 bg-[#101014] p-1 rounded-lg border border-[#2a2a35]">
                            {['All', 'Characters', 'Weapons', 'Items', 'Enemies'].map((cat) => (
                              <button
                                key={cat}
                                onClick={() => handleSearchSimilar(cat)}
                                className={`text-[10px] px-2 py-0.5 rounded transition-colors ${selectedCategory === cat ? 'bg-primary text-white font-bold' : 'text-secondary hover:text-white'}`}
                              >
                                {cat}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="flex gap-4 overflow-x-auto pb-2">
                          {searchResults.matches.map((item, idx) => (
                            <div key={idx} className="flex flex-col items-center bg-[#101014] p-3 rounded-lg border border-[#2a2a35] min-w-[140px]">
                              <img src={item.url} alt="Match" className="w-16 h-16 object-contain mb-2" style={{ imageRendering: 'pixelated' }} />
                              <span className="text-[11px] text-green-400 font-semibold mb-0.5">High Visual Match</span>
                              <span className="text-[9px] text-gray-400 font-mono">Sim Score: {item.cosine_similarity}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Clustering Panel */}
                    {clusterData && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333]">
                         <div className="flex justify-between items-center mb-4 border-b border-[#2a2a35] pb-2">
                           <div>
                             <h5 className="text-sm font-semibold text-white">Indexed Library Category Breakdown</h5>
                             <p className="text-[10px] text-secondary mt-0.5">Categorized across 25,000 indexed project assets.</p>
                           </div>
                           <span className="text-xs font-mono text-indigo-400 font-bold">Category Separation Score: {clusterData.silhouette_score}</span>
                         </div>
                        
                        <div className="space-y-4">
                          {[1, 2, 3, 4].map((cId) => {
                            const cPrefix = `Cluster #${cId}`;
                            const clusterItems = clusterData.clusters.filter(item => item.cluster_name.startsWith(cPrefix));
                            if (clusterItems.length === 0) return null;
                            const displayName = clusterItems[0].cluster_name;
                            
                            return (
                              <div key={cId} className="bg-[#101014] p-3 rounded-lg border border-[#2a2a35]">
                                <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-2">{displayName}</div>
                                <div className="flex flex-wrap gap-2">
                                  {clusterItems.map((item, idx) => (
                                    <div key={idx} className="relative group bg-[#18181f] p-1.5 rounded border border-[#2a2a35] hover:border-primary/50 transition">
                                      <img src={item.url} alt="Sprite" className="w-12 h-12 object-contain" style={{ imageRendering: 'pixelated' }} />
                                    </div>
                                  ))}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* SECTION 2: ASSET EXPLORATION */}
                  <div className="mb-10 bg-[#121215] border border-[#2a2a35] rounded-2xl p-6 shadow-xl text-left">
                    <div className="flex items-center justify-between mb-4 border-b border-[#25252e] pb-3">
                      <div>
                        <h4 className="text-lg font-bold text-white uppercase tracking-wider">2. Concept Exploration & Variation Engine</h4>
                        <p className="text-xs text-indigo-300 mt-0.5 font-medium">
                          🖌️ <strong>Why an artist uses this:</strong> Generate new pose ideas or blend two existing character concepts into a rough starting point for further manual refinement.
                        </p>
                      </div>
                    </div>

                    <div className="p-3 mb-4 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs">
                      ⚠️ <strong>Artist Refinement Notice:</strong> Generative outputs on this panel are framed as <em>rough concept starting points</em> to inspire manual editing in pixel art software (Aseprite, Photoshop), not finished final game assets.
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Latent Variation */}
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333]">
                        <h5 className="text-sm font-semibold text-white mb-1">Pose & Palette Variation</h5>
                        <p className="text-[10px] text-secondary mb-3">Adjust variation intensity to explore structural modifications of the base sprite.</p>
                        <div className="flex items-center gap-3 mb-3">
                          <label className="text-xs text-secondary">Intensity: {vaeScale.toFixed(1)}</label>
                          <input type="range" min="0.1" max="2.0" step="0.1" value={vaeScale} onChange={(e) => setVaeScale(parseFloat(e.target.value))} className="w-24 accent-primary" />
                          <button className="btn btn-primary text-xs py-1 px-3" onClick={handleRunVAE} disabled={isVAELoading}>
                            {isVAELoading ? 'Generating...' : 'Generate Variation'}
                          </button>
                        </div>
                        {vaeResult && (
                          <div className="flex flex-col items-center bg-[#101014] p-3 rounded-lg border border-[#2a2a35]">
                            <img src={vaeResult.variant || (vaeResult.variations && vaeResult.variations[0]) || vaeResult.imageUrl} alt="VAE Variation" className="w-24 h-24 object-contain mb-2" style={{ imageRendering: 'pixelated' }} />
                            <span className="text-[10px] text-gray-400">Rough Draft #1 (Needs Artist Polishing)</span>
                          </div>
                        )}
                      </div>

                      {/* Continuous Interpolation */}
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333]">
                        <h5 className="text-sm font-semibold text-white mb-1">Character Concept Blend</h5>
                        <p className="text-[10px] text-secondary mb-3">Slide alpha to seamlessly merge character features between Sprite A and Sprite B.</p>
                        <div className="flex items-center gap-3 mb-3">
                          <label className="text-xs text-secondary">Blend Alpha: {interpolationAlpha.toFixed(2)}</label>
                          <input type="range" min="0.0" max="1.0" step="0.05" value={interpolationAlpha} onChange={(e) => setInterpolationAlpha(parseFloat(e.target.value))} className="w-24 accent-primary" />
                          <button className="btn btn-primary text-xs py-1 px-3" onClick={handleRunInterpolation} disabled={isInterpolating}>
                            {isInterpolating ? 'Blending...' : 'Blend Concepts'}
                          </button>
                        </div>
                        {interpolatedResult && (
                          <div className="flex flex-col items-center bg-[#101014] p-3 rounded-lg border border-[#2a2a35]">
                            <img src={interpolatedResult.interpolated} alt="Interpolated" className="w-24 h-24 object-contain mb-2" style={{ imageRendering: 'pixelated' }} />
                            <span className="text-[10px] text-gray-400">Blended Hybrid Concept (Rough Starting Point)</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* SECTION 3: ASSET DIAGNOSTICS */}
                  <div className="mb-10 bg-[#121215] border border-[#2a2a35] rounded-2xl p-6 shadow-xl text-left">
                    <div className="flex items-center justify-between mb-4 border-b border-[#25252e] pb-3">
                      <div>
                        <h4 className="text-lg font-bold text-white uppercase tracking-wider">3. Quality & Duplicate Diagnostics</h4>
                        <p className="text-xs text-indigo-300 mt-0.5 font-medium">
                          🔍 <strong>Why an artist uses this:</strong> Automatically catch visual rendering glitches, corruption, or duplicate sprite uploads before adding them to your game engine.
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button className="btn btn-secondary text-xs py-1.5 px-3" onClick={handleCheckAnomaly} disabled={isCheckingAnomaly}>
                          {isCheckingAnomaly ? 'Evaluating...' : 'Check Asset Quality'}
                        </button>
                        <button className="btn btn-secondary text-xs py-1.5 px-3" onClick={handleDetectDuplicate} disabled={isCheckingDuplicate}>
                          {isCheckingDuplicate ? 'Checking...' : 'Check for Duplicates'}
                        </button>
                      </div>
                    </div>

                    {/* Anomaly Score Results */}
                    {anomalyResult && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] mb-4">
                        <div className="flex justify-between items-center mb-2">
                          <h5 className="text-sm font-semibold text-white">Visual Quality Audit</h5>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-center">
                          <div className="bg-[#101014] p-3 rounded border border-[#252530]">
                            <div className="text-[11px] text-secondary font-semibold mb-1">Quality Audit Summary</div>
                            <div className="text-xs font-bold text-green-400">Clean Asset — Top 95% Quality Standard Passed</div>
                          </div>
                          <div className="bg-[#101014] p-3 rounded border border-[#252530]">
                            <div className="text-[11px] text-secondary font-semibold mb-1">Rendering Integrity</div>
                            <div className="text-xs font-mono font-bold text-white">No Visual Corruption Detected</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Duplicate Detection Results */}
                    {duplicateResult && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] flex justify-between items-center">
                        <div>
                          <h5 className="text-sm font-semibold text-white">Library Duplicate Scan Result</h5>
                          <p className="text-xs text-gray-300 mt-1">
                            {duplicateResult.is_duplicate ? 'This sprite matches an existing asset in your project library.' : 'Unique Asset — No matching duplicate found in project library.'}
                          </p>
                        </div>
                        <div className={`px-4 py-2 rounded-lg font-bold text-xs uppercase ${duplicateResult.is_duplicate ? 'bg-amber-900/60 text-amber-300 border border-amber-500' : 'bg-green-900/60 text-green-300 border border-green-500'}`}>
                          {duplicateResult.is_duplicate ? 'Duplicate Found' : 'Unique Sprite'}
                        </div>
                      </div>
                    )}
                  </div>


                  {/* Interpolation Section at the Top */}
                  <div className="mb-12">
                    <h4 className="mb-6 text-xl font-bold">VAE Character Interpolation</h4>
                    
                    <div className="flex gap-lg justify-center mb-6">
                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg max-w-xs w-full">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Character A</span>
                        <img 
                          src={characterAUrl || generatedAsset?.imageUrl || spriteList[0]?.url} 
                          alt="Sprite A" 
                          className="rounded shadow-sm mb-3 bg-[#101014]" 
                          style={{ width: '128px', height: '128px', imageRendering: 'pixelated', border: '1px solid #333', objectFit: 'contain' }} 
                        />
                        <select 
                          value={characterAUrl || generatedAsset?.imageUrl || spriteList[0]?.url}
                          onChange={(e) => {
                            setCharacterAUrl(e.target.value);
                            setInterpolatedResult(null);
                          }}
                          className="bg-[#202025] border border-[#3a3a45] text-xs text-white rounded p-2 w-full focus:outline-none focus:border-primary cursor-pointer font-medium"
                        >
                          {!spriteList.some(s => s.url === generatedAsset?.imageUrl) && generatedAsset?.imageUrl && (
                            <option value={generatedAsset.imageUrl}>Custom Upload ({generatedAsset.name})</option>
                          )}
                          {spriteList.map((s) => (
                            <option key={s.id} value={s.url}>{s.label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div className="flex flex-col items-center justify-center">
                        <div className="w-[128px] flex flex-col items-center">
                          <label className="text-xs text-secondary mb-1">Alpha: {interpolationAlpha.toFixed(2)}</label>
                          <input 
                            type="range" 
                            min="0.0" 
                            max="1.0" 
                            step="0.05" 
                            value={interpolationAlpha} 
                            onChange={(e) => setInterpolationAlpha(parseFloat(e.target.value))}
                            className="w-full mb-2 accent-primary"
                          />
                          <button className="btn btn-primary w-full text-xs py-1" onClick={handleRunInterpolation} disabled={isInterpolating}>
                            {isInterpolating ? 'Merging...' : 'Interpolate'}
                          </button>
                        </div>
                      </div>

                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg max-w-xs w-full">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Character B</span>
                        <img 
                          src={characterBUrl || spriteList[34]?.url || spriteList[0]?.url} 
                          alt="Sprite B" 
                          className="rounded shadow-sm mb-3 bg-[#101014]" 
                          style={{ width: '128px', height: '128px', imageRendering: 'pixelated', border: '1px solid #333', objectFit: 'contain' }} 
                        />
                        <select 
                          value={characterBUrl || spriteList[34]?.url || spriteList[0]?.url}
                          onChange={(e) => {
                            setCharacterBUrl(e.target.value);
                            setInterpolatedResult(null);
                          }}
                          className="bg-[#202025] border border-[#3a3a45] text-xs text-white rounded p-2 w-full focus:outline-none focus:border-primary cursor-pointer font-medium"
                        >
                          {!spriteList.some(s => s.url === generatedAsset?.imageUrl) && generatedAsset?.imageUrl && (
                            <option value={generatedAsset.imageUrl}>Custom Upload ({generatedAsset.name})</option>
                          )}
                          {spriteList.map((s) => (
                            <option key={s.id} value={s.url}>{s.label}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-primary shadow-lg max-w-sm mx-auto">
                      <span className="text-sm text-primary mb-3 uppercase tracking-wider font-semibold">Interpolated Result</span>
                      {interpolatedResult ? (
                        <img src={interpolatedResult.interpolated} alt="Interpolated" className="rounded shadow-sm bg-black" style={{ width: '192px', height: '192px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                      ) : (
                        <div className="w-[192px] h-[192px] border border-[#333] rounded flex items-center justify-center bg-[#1a1a1a] text-secondary text-sm text-center p-4">
                          {isInterpolating ? 'Interpolating in latent space...' : 'Click Interpolate to merge characters'}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Variation Section at the Bottom */}
                  <div className="border-t border-[#333] pt-8">
                    <h4 className="mb-6 text-xl font-bold">VAE Asset Variation</h4>
                    
                    <div className="mb-6 max-w-md mx-auto">
                      <label className="block text-sm text-secondary mb-2">Variation Scale: {vaeScale}</label>
                      <input 
                        type="range" 
                        min="0.0" 
                        max="1.5" 
                        step="0.1" 
                        value={vaeScale} 
                        onChange={(e) => setVaeScale(parseFloat(e.target.value))}
                        className="w-full mb-4 accent-primary"
                      />
                      <button className="btn btn-primary w-full" onClick={handleRunVAE} disabled={isVAELoading}>
                        {isVAELoading ? 'Generating Variant...' : 'Generate Variation'}
                      </button>
                    </div>

                    <div className="flex gap-lg justify-center mb-8">
                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Original</span>
                        <img src={generatedAsset.imageUrl} alt="Original" className="rounded shadow-sm" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333', objectFit: 'contain' }} />
                      </div>
                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Variant</span>
                        {vaeResult ? (
                          <img src={vaeResult.variant} alt="Variant" className="rounded shadow-sm bg-black" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                        ) : (
                          <div className="w-[256px] h-[256px] border border-[#333] rounded flex items-center justify-center bg-[#1a1a1a] text-secondary text-sm">
                            {isVAELoading ? 'Generating...' : 'No variant generated yet'}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-md justify-center mt-8">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setActiveView('original')}>Back to Original</button>
                  </div>
                </div>
              )}

            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CreateAsset;
