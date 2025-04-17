import React from 'react';

/**
 * LoadingSpinner component
 * 
 * @param {Object} props Component properties
 * @param {string} props.text Optional text to display with spinner
 * @param {string} props.size Size variant: 'sm', 'md', 'lg' (default: 'md')
 * @param {boolean} props.inline Whether to display as an inline component
 */
const LoadingSpinner = ({ text, size = 'md', inline = false }) => {
  // Size class mapping
  const sizeClasses = {
    'xs': 'h-4 w-4 border-t-1 border-b-1',
    'sm': 'h-6 w-6 border-t-1 border-b-1',
    'md': 'h-12 w-12 border-t-2 border-b-2',
    'lg': 'h-16 w-16 border-t-3 border-b-3'
  };
  
  // If inline, display without text in a row layout
  if (inline) {
    return (
      <div className="inline-flex items-center">
        <div className={`animate-spin rounded-full ${sizeClasses[size] || sizeClasses.md} border-ufc-red`}></div>
        {text && <p className="text-gray-400 ml-2">{text}</p>}
      </div>
    );
  }
  
  // Default column layout with text below
  return (
    <div className="flex flex-col items-center justify-center py-4">
      <div className={`animate-spin rounded-full ${sizeClasses[size] || sizeClasses.md} border-ufc-red mb-2`}></div>
      {text && <p className="text-gray-400 text-sm">{text}</p>}
    </div>
  );
};

export default LoadingSpinner;