/**
 * GameForge AI — Character Pose Deformation Studio (Root Export Wrapper)
 * Re-exports the modular PoseDeformationStudio container from components/deformation/.
 */

import PoseDeformationStudio from './deformation/PoseDeformationStudio';

export default PoseDeformationStudio;
export { default as PoseSelector } from './deformation/PoseSelector';
export { default as JointEditor } from './deformation/JointEditor';
export { default as DeformationControls } from './deformation/DeformationControls';
export { default as DeformationPreview } from './deformation/DeformationPreview';
