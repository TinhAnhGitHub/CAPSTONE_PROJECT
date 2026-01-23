import React from 'react'

export default function IngestedStatus({ percentage }) {
  const radius = 18;
  const circumference = 2 * Math.PI * radius;
  const progress = ((percentage || 0) / 100) * circumference;
  const dashOffset = circumference - progress;

  if (percentage === 100 || percentage === undefined) return null;

  return (
    <div className="relative flex items-center justify-center">
      {/* Background overlay */}
      {/* fix temporary */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm rounded-lg" style={{ margin: '-300%' }} />

      {/* Circular progress */}
      <div className="relative w-14 h-14 flex items-center justify-center">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 44 44">
          {/* Background circle */}
          <circle
            cx="22"
            cy="22"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            className="text-white/20"
          />
          {/* Progress circle */}
          <circle
            cx="22"
            cy="22"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className="text-accent transition-all duration-300"
          />
        </svg>
        {/* Percentage text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-medium text-white">{Math.round(percentage)}%</span>
        </div>
      </div>
    </div>
  )
}
