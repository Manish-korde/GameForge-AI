import React, { useRef, useState } from 'react';

// Canonical bone connections matching backend/deformation/skeleton.py
export const BONE_CONNECTIONS = [
  ['head', 'neck'],
  ['neck', 'torso'],
  ['torso', 'pelvis'],
  ['neck', 'left_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['neck', 'right_shoulder'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['pelvis', 'left_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['pelvis', 'right_hip'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle']
];

/**
 * Helper to safely extract a joint by canonical name from either Dict or Array representation.
 */
export const getJointByName = (skel, name) => {
  if (!skel || !skel.joints) return null;
  if (Array.isArray(skel.joints)) {
    return skel.joints.find(j => j.name === name) || null;
  }
  return skel.joints[name] || null;
};

/**
 * Helper to safely extract an iterable list of joints.
 */
export const getJointList = (skel) => {
  if (!skel || !skel.joints) return [];
  if (Array.isArray(skel.joints)) return skel.joints;
  return Object.values(skel.joints);
};

const getBoneStroke = (jointA, jointB) => {
  if (jointA.includes('left') || jointB.includes('left')) return '#34d399'; // Emerald
  if (jointA.includes('right') || jointB.includes('right')) return '#c084fc'; // Purple
  return '#21d4fd'; // Cyan spine
};

/**
 * JointEditor component: renders interactive SVG skeleton over sprite canvas with draggable joints.
 */
const JointEditor = ({
  imageUrl,
  skeleton,
  isSkeletonVisible = true,
  activeJoint,
  onSelectJoint,
  onJointChange
}) => {
  const [draggingJoint, setDraggingJoint] = useState(null);
  const svgRef = useRef(null);

  const handleMouseDownJoint = (e, jointName) => {
    e.stopPropagation();
    setDraggingJoint(jointName);
    if (onSelectJoint) onSelectJoint(jointName);
  };

  const handleMouseMove = (e) => {
    if (!draggingJoint || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const rawX = (e.clientX - rect.left) / rect.width;
    const rawY = (e.clientY - rect.top) / rect.height;

    const clampedX = Math.max(0.02, Math.min(0.98, rawX));
    const clampedY = Math.max(0.02, Math.min(0.98, rawY));

    if (onJointChange) {
      onJointChange(draggingJoint, Number(clampedX.toFixed(3)), Number(clampedY.toFixed(3)));
    }
  };

  const handleMouseUp = () => {
    setDraggingJoint(null);
  };

  return (
    <div
      className="pose-canvas-wrapper"
      style={{ width: '256px', height: '256px' }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <img src={imageUrl} alt="Source Sprite" className="pose-sprite-img" />

      {/* SVG Skeleton Overlay */}
      {isSkeletonVisible && skeleton && (
        <svg ref={svgRef} className="pose-skeleton-svg" viewBox="0 0 1 1">
          {/* Bone Connections */}
          {BONE_CONNECTIONS.map(([nameA, nameB], idx) => {
            const jointA = getJointByName(skeleton, nameA);
            const jointB = getJointByName(skeleton, nameB);
            if (!jointA || !jointB) return null;
            return (
              <line
                key={`bone-${idx}`}
                x1={jointA.x}
                y1={jointA.y}
                x2={jointB.x}
                y2={jointB.y}
                className="pose-bone-line"
                stroke={getBoneStroke(nameA, nameB)}
              />
            );
          })}

          {/* Joint Nodes */}
          {getJointList(skeleton).map(j => (
            <circle
              key={`joint-${j.name}`}
              cx={j.x}
              cy={j.y}
              r={activeJoint === j.name ? 0.025 : 0.018}
              className="pose-joint-node"
              fill={activeJoint === j.name ? '#f59e0b' : '#ffffff'}
              stroke="#0b0d12"
              strokeWidth="0.005"
              onMouseDown={e => handleMouseDownJoint(e, j.name)}
            >
              <title>{j.name} ({j.x.toFixed(2)}, {j.y.toFixed(2)})</title>
            </circle>
          ))}
        </svg>
      )}
    </div>
  );
};

export default JointEditor;
