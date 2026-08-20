import { useState } from 'react';
import { generateDesign, generateImage, runAEInference } from '../services/api';

const CreateAsset = () => {
  const [prompt, setPrompt] = useState('');
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [designSpec, setDesignSpec] = useState(null);
  const [generatedAsset, setGeneratedAsset] = useState(null);
  const [aeResult, setAeResult] = useState(null);
  const [isAELoading, setIsAELoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

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
    setIsLoading(false);
    setStep(3);
  };

  const handleRunAE = async () => {
    if (!generatedAsset) return;
    setIsAELoading(true);
    try {
      const result = await runAEInference(generatedAsset.imageUrl);
      setAeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run AE Inference");
    } finally {
      setIsAELoading(false);
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
    const localUrl = URL.createObjectURL(file);
    setGeneratedAsset({
      id: 'custom-' + Date.now(),
      name: file.name,
      type: 'Image',
      source: 'Upload',
      model: 'Custom Upload',
      prompt: 'Custom user uploaded image for AE testing',
      imageUrl: localUrl,
      createdAt: new Date().toISOString(),
      status: 'Completed'
    });
    setAeResult(null);
    setStep(3);
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
            <h3 className="mb-4">GAME CONCEPT</h3>
            <div className="grid grid-cols-2 gap-md mb-4">
              <div>
                <span className="text-sm text-secondary block">Theme</span>
                {designSpec.theme}
              </div>
              <div>
                <span className="text-sm text-secondary block">Environment</span>
                {designSpec.environment}
              </div>
              <div>
                <span className="text-sm text-secondary block">Characters</span>
                {designSpec.characters.length}
              </div>
              <div>
                <span className="text-sm text-secondary block">Visual Style</span>
                {designSpec.visualStyle}
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
             <pre className="raw-output">{designSpec.rawOutput}</pre>
          </div>
        </div>
      )}

      {step === 3 && generatedAsset && (
        <div className="grid grid-cols-3 gap-lg mt-4">
          <div className="col-span-1">
             <div className="card">
               <h3 className="mb-4">Asset Details</h3>
               <div className="text-sm text-secondary mb-1">Name</div>
               <div className="mb-4">{generatedAsset.name}</div>
               <div className="text-sm text-secondary mb-1">Prompt</div>
               <div className="mb-4 text-sm">{generatedAsset.prompt}</div>
               <div className="text-sm text-secondary mb-1">Generated by</div>
               <div className="mb-4">{generatedAsset.model}</div>
               <button className="btn btn-secondary w-full mb-2" onClick={handleRunAE} disabled={isAELoading}>
                 {isAELoading ? 'Running AE...' : 'Run through AE'}
               </button>
               <button className="btn btn-secondary w-full mb-2">Create VAE Variations</button>
             </div>
          </div>
          <div className="col-span-2">
            <div className="card text-center flex flex-col items-center h-full">
              {!aeResult ? (
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
              ) : (
                <div className="w-full">
                  <h4 className="mb-6 text-xl font-bold">Autoencoder Inference</h4>
                  <div className="flex gap-lg justify-center mb-8">
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
                    <div className="bg-[#151515] rounded-xl p-6 border border-border mb-8 max-w-2xl mx-auto shadow-lg">
                      <h5 className="text-sm text-secondary uppercase tracking-wider mb-4 font-semibold text-left">Real-Time Evaluation Metrics</h5>
                      
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

                      {/* Silent Footer Metrics */}
                      <div className="flex justify-between items-center text-[#555] text-[10px] border-t border-[#333] pt-3 px-2">
                        <span>PSNR: {aeResult.metrics.psnr.toFixed(2)} dB</span>
                        <span>SSIM: {aeResult.metrics.ssim.toFixed(4)}</span>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-md justify-center">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setAeResult(null)}>Back to Original</button>
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
