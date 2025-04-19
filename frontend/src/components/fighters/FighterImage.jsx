import React, { useState, useEffect, useRef } from 'react';
import LoadingSpinner from '../common/LoadingSpinner';
import { fighterService } from '../../services/api';

/**
 * Fighter image component with lazy loading, intersection observer, and multi-source image fetching
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
      console.log(`Using provided image source for ${alt}`);
      setImageSrc(src);
      setLoading(false);
      return;
    }
    
    // No src or placeholder src - need to fetch with multi-source fallback
    try {
      setFetchingImage(true);
      
      // First check localStorage cache directly to avoid race conditions
      const cacheKey = `fighter_image_${alt.toLowerCase().replace(/\s+/g, '_')}`;
      const cachedImage = localStorage.getItem(cacheKey);
      
      if (cachedImage && cachedImage !== "/static/placeholder.png") {
        console.log(`Using cached image for ${alt}`);
        setImageSrc(cachedImage);
        setLoading(false);
        setFetchingImage(false);
        return;
      }
      
      // If not in cache, try to get image without specifying a source
      console.log(`Fetching image for ${alt} with automatic source selection`);
      let imageUrl = await fighterService.getFighterImage(alt);
      
      if (imageUrl && imageUrl !== "/static/placeholder.png") {
        console.log(`Found image for ${alt} from automatic source selection`);
        // Store in localStorage to avoid future API calls
        try {
          localStorage.setItem(cacheKey, imageUrl);
        } catch (e) {
          // Ignore errors setting cache
        }
        
        setImageSrc(imageUrl);
        setLoading(false);
        setFetchingImage(false);
        return;
      }
      
      // If the automatic selection didn't work, try each source specifically
      // This will bypass any cached "not found" results
      const sources = ['wikipedia', 'sherdog', 'ufc', 'google'];
      for (const source of sources) {
        try {
          console.log(`Explicitly trying ${source} for ${alt} image...`);
          imageUrl = await fighterService.getFighterImage(alt, source);
          
          if (imageUrl && imageUrl !== "/static/placeholder.png") {
            console.log(`Successfully fetched ${alt} image from ${source}`);
            
            // Update localStorage cache with this successful image
            try {
              localStorage.setItem(cacheKey, imageUrl);
            } catch (e) {
              // Ignore errors setting cache
            }
            
            setImageSrc(imageUrl);
            setLoading(false);
            setFetchingImage(false);
            return;
          }
        } catch (sourceErr) {
          console.warn(`Failed to fetch from ${source}:`, sourceErr);
          // Continue to next source
        }
      }
      
      // If we get here, all sources truly failed
      console.warn(`All image sources failed for ${alt}, using placeholder`);
      setImageSrc(placeholderImage);
      setError(true);
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
      // Always check caching and do proper image loading
      // regardless of src to ensure consistent behavior
      fetchFighterImage();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible, src, alt]);
  
  // Handle image loading success
  const handleLoad = () => {
    // Only update if imageSrc is not the placeholder
    if (imageSrc && imageSrc !== placeholderImage) {
      setLoading(false);
      setError(false);
      
      // Store successful image in cache to prevent future issues
      if (alt) {
        const cacheKey = `fighter_image_${alt.toLowerCase().replace(/\s+/g, '_')}`;
        try {
          localStorage.setItem(cacheKey, imageSrc);
        } catch (e) {
          // Ignore errors setting cache
        }
      }
    } else if (imageSrc === placeholderImage) {
      // If displaying placeholder, mark as error
      setLoading(false);
      setError(true);
    }
  };
  
  // Handle image loading error
  const handleError = () => {
    console.error(`Image load error for ${alt}`);
    setLoading(false);
    setError(true);
    setImageSrc(placeholderImage);
    
    // Clear invalid image URL from cache
    if (alt) {
      const cacheKey = `fighter_image_${alt.toLowerCase().replace(/\s+/g, '_')}`;
      try {
        // Only remove if it's not already the placeholder
        const current = localStorage.getItem(cacheKey);
        if (current && current !== "/static/placeholder.png") {
          localStorage.removeItem(cacheKey);
        }
      } catch (e) {
        // Ignore cache errors
      }
    }
  };

  // Enhanced size class mapping with MUCH larger size options 
  // Using proper aspect ratio (460x700 ~= 2:3) to ensure heads are not cut off
  const sizeClasses = {
    'xs': 'w-16 h-24',      // Small thumbnails
    'sm': 'w-32 h-48',      // Small cards
    'md': 'w-60 h-80',      // Standard card size - INCREASED for better visibility
    'lg': 'w-72 h-96',      // Large displays - INCREASED
    'xl': 'w-80 h-120',     // Extra large displays
    '2xl': 'w-96 h-144'     // Maximum size for headers
  };
  
  // Display spinner if image is still loading or being fetched when visible
  const isLoading = isVisible && (loading || fetchingImage);
  
  // Determine if we should use rounded styling
  // Only use circular styling for placeholders, NEVER for real fighter images
  const shouldBeRounded = rounded !== undefined 
    ? rounded 
    : (!imageSrc || imageSrc === placeholderImage || error);
  
  // Compose CSS classes with proper object-fit settings
  // object-contain ensures the entire fighter is visible without cropping
  const imageClasses = `
    ${sizeClasses[size] || sizeClasses.md}
    ${shouldBeRounded ? 'rounded-full' : 'rounded-md'}
    ${withBorder ? `border-2 ${borderColor}` : ''}
    object-contain
    object-top
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