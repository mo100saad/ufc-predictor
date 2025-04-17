import React, { useState, useEffect, useRef } from 'react';
import LoadingSpinner from '../common/LoadingSpinner';
import { fighterService } from '../../services/api';

/**
 * Reusable fighter image component with lazy loading, intersection observer,
 * and on-demand image fetching. Enhanced to display larger, properly scaled images.
 * 
 * @param {Object} props Component properties
 * @param {string} props.src The source URL of the fighter image
 * @param {string} props.alt The alt text for the image
 * @param {string} props.className Additional CSS classes for styling
 * @param {string} props.size Size variant: 'xs', 'sm', 'md', 'lg', 'xl', '2xl' (default: 'md')
 * @param {boolean} props.rounded Whether to apply rounded styling (default: false for valid images, true for placeholders)
 * @param {boolean} props.withBorder Whether to apply border styling (default: true)
 * @param {string} props.borderColor CSS color for the border (default: 'border-gray-600')
 */
const FighterImage = ({ 
  src, 
  alt, 
  className = '', 
  size = 'md', 
  rounded, 
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

  // Initialize image source to null (instead of placeholder) to show loading state
  useEffect(() => {
    // Don't show placeholder immediately, show loading state instead
    setImageSrc(null);
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

  // Enhanced size class mapping with MUCH larger size options and fixed height-to-width ratio for fighter images
  // Using a taller aspect ratio (460x700 ~= 2:3) to ensure heads are not cut off
  const sizeClasses = {
    'xs': 'w-16 h-24',     // 1:1.5 ratio - better for profile pictures
    'sm': 'w-32 h-48',     // 1:1.5 ratio - for fighter cards
    'md': 'w-48 h-72',     // 1:1.5 ratio - good general size
    'lg': 'w-64 h-96',     // 1:1.5 ratio - for prediction display
    'xl': 'w-80 h-120',    // 1:1.5 ratio - extra large display
    '2xl': 'w-96 h-144'    // 1:1.5 ratio - max size for headers
  };
  
  // Display spinner if image is still loading or being fetched when visible
  const isLoading = isVisible && (loading || fetchingImage);
  
  // Determine if we should use rounded styling
  // For valid fighter images: no rounding, for placeholders: use circular shape
  const shouldBeRounded = rounded !== undefined 
    ? rounded 
    : (!imageSrc || imageSrc === placeholderImage || error);
  
  // Compose CSS classes with conditional rounded styling and improved sizing for large images
  const imageClasses = `
    ${sizeClasses[size] || sizeClasses.md}
    ${shouldBeRounded ? 'rounded-full' : 'rounded-md'}
    ${withBorder ? `border-2 ${borderColor}` : ''}
    object-contain
    !object-top
    max-h-full
    max-w-full
    transition-all duration-300
    ${isLoading ? 'opacity-50' : 'opacity-100'}
    ${className}
  `;
  
  return (
    <div 
      ref={imgRef} 
      className={`relative ${sizeClasses[size] || sizeClasses.md} 
        ${shouldBeRounded ? 'rounded-full' : 'rounded-md'}`}
    >
      {/* Enhanced loading indicator - show whenever loading or fetching, with better visibility */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-800/70 z-10 backdrop-blur-sm">
          <div className="flex flex-col items-center">
            <LoadingSpinner size="md" />
            <span className="text-xs text-gray-300 mt-2">Loading fighter image...</span>
          </div>
        </div>
      )}
      
      {/* Image placeholder immediately or actual image when loaded - properly scaled */}
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