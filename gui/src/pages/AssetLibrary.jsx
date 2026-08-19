import { useState, useEffect } from 'react';
import { fetchLibraryAssets } from '../services/api';

const AssetLibrary = () => {
  const [assets, setAssets] = useState([]);
  
  useEffect(() => {
    fetchLibraryAssets().then(res => setAssets(res.data));
  }, []);

  return (
    <div className="asset-library">
      <h2>Asset Library</h2>
      <div className="flex gap-sm mt-4 mb-6">
        <button className="btn btn-primary text-sm py-1">All</button>
        <button className="btn btn-secondary text-sm py-1">Characters</button>
        <button className="btn btn-secondary text-sm py-1">Weapons</button>
        <button className="btn btn-secondary text-sm py-1">Environments</button>
      </div>

      <div className="grid grid-cols-4 gap-md">
        {assets.map(asset => (
          <div key={asset.id} className="card asset-card p-0 overflow-hidden">
            <img src={asset.imageUrl} alt={asset.name} className="w-full h-48 object-cover" />
            <div className="p-4">
              <h4 className="m-0 text-md">{asset.name}</h4>
              <div className="text-xs text-secondary mt-1">{asset.type} • {asset.model}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AssetLibrary;
