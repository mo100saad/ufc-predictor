import React, { useState, useEffect, useRef } from 'react';
import LoadingSpinner from '../common/LoadingSpinner';
import { fighterService } from '../../services/api';

/**
 * Reusable fighter image component with lazy loading, intersection observer,
 * and on-demand image fetching.
 * 
 * @param {Object} props Component properties
 * @param {string} props.src The source URL of the fighter image
 * @param {string} props.alt The alt text for the image
 * @param {string} props.className Additional CSS classes for styling
 * @param {string} props.size Size variant: 'xs', 'sm', 'md', 'lg', 'xl' (default: 'md')
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
  const [imageSrc, setImageSrc] = useState(null);
  const [fetchingImage, setFetchingImage] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const imgRef = useRef(null);
  
  // Placeholder image (fighter silhouette SVG)
  const placeholderImage = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNkI3MjgwIiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS11c2VyIj48cGF0aCBkPSJNMTkgMjFhNyA3IDAgMCAwLTE0IDAiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEwIiByPSI0Ii8+PC9zdmc+';

  // Set placeholder immediately on first render
  useEffect(() => {
    setImageSrc(placeholderImage);
  }, []);

  // Set up intersection observer for lazy loading
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        // Mark as visible once the element enters viewport
        if (entry.isIntersecting) {
          setIsVisible(true);
          // Disconnect observer after it becomes visible
          observer.disconnect();
        }
      },
      {
        rootMargin: '200px', // Start loading when within 200px of viewport
        threshold: 0.1       // Trigger when at least 10% is visible
      }
    );
    
    if (imgRef.current) {
      observer.observe(imgRef.current);
    }
    
    return () => {
      if (imgRef.current) {
        observer.unobserve(imgRef.current);
      }
    };
  }, []);

  // Function to fetch fighter image if needed
  const fetchFighterImage = async () => {
    if (!alt || fetchingImage || !isVisible) return;
    
    // Already have source from props and it's not a placeholder
    if (src && src !== "/static/placeholder.png") {
      setImageSrc(src);
      return;
    }
    
    // No src or placeholder src - need to fetch
    try {
      setFetchingImage(true);
      const imageUrl = await fighterService.getFighterImage(alt);
      if (imageUrl && imageUrl !== "/static/placeholder.png") {
        setImageSrc(imageUrl);
      } else {
        setImageSrc(placeholderImage);
        setError(true);
      }
    } catch (err) {
      console.error("Error fetching fighter image:", err);
      setImageSrc(placeholderImage);
      setError(true);
    } finally {
      setFetchingImage(false);
    }
  };
  
  // Only fetch the image when the component becomes visible
  useEffect(() => {
    if (isVisible) {
      // If src is provided and valid, use it directly
      if (src && src !== "/static/placeholder.png") {
        setImageSrc(src);
      } 
      // If no src is provided or it's a placeholder, try to fetch it
      else if (alt) {
        fetchFighterImage();
      }
    }
  }, [src, alt, isVisible]);
  
  // Handle image loading success
  const handleLoad = () => {
    setLoading(false);
    setError(false);
  };
  
  // Handle image loading error
  const handleError = () => {
    setLoading(false);
    setError(true);
    setImageSrc(placeholderImage);
  };

  // Size class mapping
  const sizeClasses = {
    'xs': 'w-8 h-8',
    'sm': 'w-12 h-12', 
    'md': 'w-16 h-16',
    'lg': 'w-24 h-24',
    'xl': 'w-32 h-32'
  };
  
  // Display spinner if image is still loading or being fetched when visible
  const isLoading = isVisible && (loading || fetchingImage);
  
  // Compose CSS classes
  const imageClasses = `
    ${sizeClasses[size] || sizeClasses.md}
    ${rounded ? 'rounded-full' : ''}
    ${withBorder ? `border-2 ${borderColor}` : ''}
    overflow-hidden
    object-cover
    transition-opacity duration-300
    ${isLoading ? 'opacity-50' : 'opacity-100'}
    ${className}
  `;
  
  return (
    <div 
      ref={imgRef} 
      className={`relative ${sizeClasses[size] || sizeClasses.md} overflow-hidden ${rounded ? 'rounded-full' : ''}`}
    >
      {/* Loading indicator - only show when actually fetching and element is visible */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
          <LoadingSpinner size="sm" />
        </div>
      )}
      
      {/* Image placeholder immediately or actual image when loaded */}
      <img
        src={imageSrc || placeholderImage}
        alt={alt || "Fighter"}
        className={imageClasses}
        onLoad={handleLoad}
        onError={handleError}
      />
    </div>
  );
};

export default FighterImage;