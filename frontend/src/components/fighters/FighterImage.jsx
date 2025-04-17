import React, { useState } from 'react';
import LoadingSpinner from '../common/LoadingSpinner';

/**
 * Reusable fighter image component with loading state and fallback
 * 
 * @param {Object} props Component properties
 * @param {string} props.src The source URL of the fighter image
 * @param {string} props.alt The alt text for the image
 * @param {string} props.className Additional CSS classes for styling
 * @param {string} props.size Size variant: 'sm', 'md', 'lg' (default: 'md')
 * @param {boolean} props.rounded Whether to apply rounded styling (default: true)
 * @param {boolean} props.withBorder Whether to apply border styling (default: true)
 * @param {string} props.borderColor CSS color for the border (default: 'border-gray-600')
 */
const FighterImage = ({ 
  src, 
  alt, 
  className = '', 
  size = 'md', 
  rounded = true, 
  withBorder = true,
  borderColor = 'border-gray-600'
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  
  // Placeholder image (fighter silhouette SVG)
  const placeholderImage = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNkI3MjgwIiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS11c2VyIj48cGF0aCBkPSJNMTkgMjFhNyA3IDAgMCAwLTE0IDAiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEwIiByPSI0Ii8+PC9zdmc+';

  // Determine actual source - use provided source or fallback
  const imageSrc = (!src || src === "/static/placeholder.png") ? placeholderImage : src;
  
  // Handle image loading success
  const handleLoad = () => {
    setLoading(false);
    setError(false);
  };
  
  // Handle image loading error
  const handleError = () => {
    setLoading(false);
    setError(true);
  };

  // Size class mapping
  const sizeClasses = {
    'xs': 'w-8 h-8',
    'sm': 'w-12 h-12', 
    'md': 'w-16 h-16',
    'lg': 'w-24 h-24',
    'xl': 'w-32 h-32'
  };
  
  // Compose CSS classes
  const imageClasses = `
    ${sizeClasses[size] || sizeClasses.md}
    ${rounded ? 'rounded-full' : ''}
    ${withBorder ? `border-2 ${borderColor}` : ''}
    overflow-hidden
    object-cover
    transition-opacity duration-300
    ${loading ? 'opacity-0' : 'opacity-100'}
    ${className}
  `;
  
  return (
    <div className={`relative ${sizeClasses[size] || sizeClasses.md} overflow-hidden ${rounded ? 'rounded-full' : ''}`}>
      {/* Loading indicator */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
          <LoadingSpinner size="sm" />
        </div>
      )}
      
      {/* Actual image */}
      <img
        src={error ? placeholderImage : imageSrc}
        alt={alt || "Fighter"}
        className={imageClasses}
        onLoad={handleLoad}
        onError={handleError}
      />
    </div>
  );
};

export default FighterImage;