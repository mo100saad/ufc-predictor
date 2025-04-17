import axios from 'axios';

// Define API URL with fallback for news API
const API_URL = '/api';
const NEWS_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:5050' : '/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fighterService = {
  getAllFighters: async () => {
    const response = await api.get('/fighters');
    return response.data.fighters;
  },
  
  getFighterByName: async (name) => {
    try {
      console.log(`Requesting fighter: ${name}`);
      const response = await api.get(`/fighters/${encodeURIComponent(name)}`);
      console.log('API Response:', response.data);
      return response.data.fighter;
    } catch (error) {
      console.error('Error fetching fighter:', error.response?.data || error.message);
      throw error;
    }
  },
  
  // Enhanced fighter image service with improved multi-source fallback
  getFighterImage: async (name, source = null) => {
    try {
      // Check if we have name
      if (!name) {
        return "/static/placeholder.png";
      }
      
      // Cache check to avoid unnecessary network requests
      const cacheKey = `fighter_image_${name.toLowerCase().replace(/\s+/g, '_')}`;
      
      // If a specific source is requested, bypass the cache completely
      if (source) {
        console.log(`Bypassing cache for ${name}, source=${source}`);
      } else {
        // Check for cached image if no specific source is requested
        const cachedImage = localStorage.getItem(cacheKey);
        if (cachedImage && cachedImage !== "/static/placeholder.png") {
          console.log(`Using cached image for ${name}`);
          return cachedImage;
        }
      }
      
      // Build API endpoint
      let endpoint = `/fighters/${encodeURIComponent(name)}/image`;
      if (source) {
        endpoint += `?source=${source}`;
      }
      
      console.log(`Requesting image for ${name} from endpoint: ${endpoint}`);
      const response = await api.get(endpoint);
      const imageUrl = response.data.image_url;
      const actualSource = response.data.source || "unknown";
      
      // If we got a valid image, save to cache and return it
      if (imageUrl && imageUrl !== "/static/placeholder.png") {
        console.log(`Found image for ${name} from ${actualSource}`);
        
        // Store in localStorage cache (unless it's a source-specific request)
        if (!source) {
          try {
            localStorage.setItem(cacheKey, imageUrl);
            console.log(`Saved image for ${name} to local cache`);
          } catch (cacheErr) {
            console.warn('Failed to cache fighter image:', cacheErr);
          }
        }
        
        return imageUrl;
      }
      
      // If no image found and no specific source requested, we already tried all sources
      if (!source) {
        console.warn(`All image sources failed for ${name}`);
        return "/static/placeholder.png";
      }
      
      // If specific source request failed, return placeholder
      console.warn(`Failed to get image from ${source} for ${name}`);
      return "/static/placeholder.png";
    } catch (error) {
      console.error('Error fetching fighter image:', error.response?.data || error.message);
      return "/static/placeholder.png";
    }
  }
};

export const fightService = {
  getAllFights: async () => {
    const response = await api.get('/fights');
    return response.data.fights;
  },
};

export const predictionService = {
  predictFight: async (fighter1, fighter2) => {
    console.log("Sending prediction request with data:", { fighter1, fighter2 });
    
    // Apply client-side weight class correction for extreme mismatches before sending to API
    const adjustedFighters = applyWeightClassCorrection(fighter1, fighter2);
    
    const response = await api.post('/predict', {
      fighter1: adjustedFighters.fighter1,
      fighter2: adjustedFighters.fighter2
    });
    
    // Get prediction from server
    let prediction = response.data.prediction;
    
    // Apply additional probability adjustment for extreme weight differences
    prediction = adjustPredictionForWeightDisparity(prediction, fighter1, fighter2);
    
    return prediction;
  },
};

// Helper function to apply weight class correction to prevent unrealistic predictions
function applyWeightClassCorrection(fighter1, fighter2) {
  // Deep clone the fighters to avoid modifying originals
  const f1 = JSON.parse(JSON.stringify(fighter1));
  const f2 = JSON.parse(JSON.stringify(fighter2));
  
  // Calculate weight difference
  const weight1 = f1.weight || 0;
  const weight2 = f2.weight || 0;
  const weightDiff = Math.abs(weight1 - weight2);
  
  // If weight difference is extreme (more than 15kg / ~33lbs), apply corrections
  if (weightDiff > 15) {
    console.log(`Extreme weight difference detected: ${weightDiff.toFixed(1)}kg - applying correction`);
    
    // Determine which fighter is heavier
    const heavierFighter = weight1 > weight2 ? f1 : f2;
    const lighterFighter = weight1 > weight2 ? f2 : f1;
    
    // Apply a penalty to the lighter fighter's stats based on the weight difference
    const penaltyFactor = Math.min(0.7, Math.max(0.3, 1 - (weightDiff / 100)));
    
    // Reduce striking metrics for the lighter fighter against a much heavier opponent
    lighterFighter.SLpM = lighterFighter.SLpM * penaltyFactor;
    lighterFighter.str_acc = lighterFighter.str_acc * penaltyFactor;
    
    console.log(`Applied penalty factor ${penaltyFactor.toFixed(2)} to lighter fighter's offensive stats`);
  }
  
  return { fighter1: f1, fighter2: f2 };
}

// Helper function to adjust prediction probabilities for extreme weight disparities
function adjustPredictionForWeightDisparity(prediction, fighter1, fighter2) {
  // Guard against malformed prediction object
  if (!prediction || typeof prediction !== 'object') return prediction;
  
  const weight1 = fighter1.weight || 0;
  const weight2 = fighter2.weight || 0;
  const weightDiff = Math.abs(weight1 - weight2);
  
  // Define weight class difference thresholds
  // 10kg (~22lbs) = approximately 1 weight class
  // 20kg (~44lbs) = approximately 2 weight classes
  // 30kg (~66lbs) = approximately 3 weight classes or more
  
  if (weightDiff > 20) {
    // Clone prediction to avoid modifying original
    const adjustedPrediction = {...prediction};
    
    // Determine which fighter is heavier and adjust probabilities
    if (weight1 > weight2) {
      // Fighter 1 is heavier
      const adjustmentFactor = Math.min(0.3, weightDiff / 100);
      
      // Take probability from lighter fighter and give to heavier fighter
      adjustedPrediction.fighter2_win_probability = Math.max(0.05, 
        adjustedPrediction.fighter2_win_probability - adjustmentFactor);
      adjustedPrediction.fighter1_win_probability = Math.min(0.95, 
        1 - adjustedPrediction.fighter2_win_probability);
      
      console.log(`Adjusted prediction for weight disparity: ${adjustmentFactor.toFixed(2)} in favor of heavier fighter 1`);
    } else {
      // Fighter 2 is heavier
      const adjustmentFactor = Math.min(0.3, weightDiff / 100);
      
      // Take probability from lighter fighter and give to heavier fighter
      adjustedPrediction.fighter1_win_probability = Math.max(0.05, 
        adjustedPrediction.fighter1_win_probability - adjustmentFactor);
      adjustedPrediction.fighter2_win_probability = Math.min(0.95, 
        1 - adjustedPrediction.fighter1_win_probability);
      
      console.log(`Adjusted prediction for weight disparity: ${adjustmentFactor.toFixed(2)} in favor of heavier fighter 2`);
    }
    
    return adjustedPrediction;
  }
  
  return prediction;
}

export const csvService = {
  getCSVStatus: async () => {
    const response = await api.get('/manage-csv');
    return response.data;
  },
  
  syncCSV: async () => {
    const response = await api.post('/manage-csv', {
      action: 'sync'
    });
    return response.data;
  },
};

// Create a separate news API instance
const newsApi = axios.create({
  baseURL: NEWS_URL,
  headers: {
    'Content-Type': 'application/json',
  }
});

export const newsService = {
  getUFCNews: async () => {
    try {
      console.log(`Attempting to fetch news from ${NEWS_URL}/news`);
      
      // Try both endpoints in parallel to get the most articles
      const [mainResponse, backupResponse] = await Promise.allSettled([
        newsApi.get('/news'),
        api.get('/news')
      ]);
      
      let allArticles = [];
      
      // Process main API response if successful
      if (mainResponse.status === 'fulfilled' && 
          mainResponse.value.data && 
          mainResponse.value.data.news && 
          mainResponse.value.data.news.length > 0) {
        
        const mainArticles = mainResponse.value.data.news.map(article => ({
          ...article,
          id: article.id || article.url || Math.random().toString(36).substr(2, 9),
          imageUrl: article.imageUrl || article.urlToImage || '/static/placeholder.png',
          source: article.source?.name || article.source || 'MMA News',
          publishedAt: article.publishedAt || article.date || new Date().toISOString(),
          imageLoaded: false
        }));
        
        console.log(`Retrieved ${mainArticles.length} news articles from primary source`);
        allArticles = [...mainArticles];
      }
      
      // Process backup API response if successful
      if (backupResponse.status === 'fulfilled' && 
          backupResponse.value.data && 
          backupResponse.value.data.news && 
          backupResponse.value.data.news.length > 0) {
        
        const backupArticles = backupResponse.value.data.news.map(article => ({
          id: article.url || Math.random().toString(36).substr(2, 9),
          title: article.title || "UFC News",
          description: article.description || "Latest UFC news and updates",
          url: article.url || "https://www.ufc.com/news",
          imageUrl: article.urlToImage || article.imageUrl || '/static/placeholder.png',
          source: article.source?.name || article.source || 'MMA News',
          publishedAt: article.publishedAt || article.date || new Date().toISOString(),
          imageLoaded: false
        }));
        
        console.log(`Retrieved ${backupArticles.length} news articles from backup source`);
        
        // Merge articles, avoiding duplicates by URL
        const existingUrls = new Set(allArticles.map(a => a.url));
        const uniqueBackupArticles = backupArticles.filter(a => !existingUrls.has(a.url));
        
        allArticles = [...allArticles, ...uniqueBackupArticles];
      }
      
      // If we have articles after trying both sources, return them
      if (allArticles.length > 0) {
        // Sort by date (newest first) but don't filter by date
        return allArticles.sort((a, b) => {
          const dateA = new Date(a.publishedAt || a.date || 0);
          const dateB = new Date(b.publishedAt || b.date || 0);
          return dateB - dateA;
        });
      }
      
      // If no articles at all, try direct API call
      console.log("No articles from primary sources, trying direct API call");
      try {
        const directResponse = await axios.get(`/api/news`);
        if (directResponse.data && directResponse.data.news && directResponse.data.news.length > 0) {
          console.log(`Got ${directResponse.data.news.length} articles from direct API call`);
          
          const directArticles = directResponse.data.news.map(article => ({
            id: article.url || Math.random().toString(36).substr(2, 9),
            title: article.title || "UFC News",
            description: article.description || "Latest UFC news and updates",
            url: article.url || "https://www.ufc.com/news",
            imageUrl: article.urlToImage || article.imageUrl || '/static/placeholder.png',
            source: article.source?.name || article.source || 'MMA News',
            publishedAt: article.publishedAt || article.date || new Date().toISOString(),
            imageLoaded: false
          }));
          
          return directArticles;
        }
      } catch (directError) {
        console.error("Direct API call failed:", directError);
      }
      
      // If all else fails, return fallback content
      console.warn("All news sources failed, using fallback content");
      return generateFallbackNews();
      
    } catch (error) {
      console.error("Error fetching UFC news:", error);
      
      // Try direct fetch as last resort
      try {
        const proxyUrl = "https://api.allorigins.win/raw?url=" + 
          encodeURIComponent("https://www.ufc.com/news");
        
        const response = await fetch(proxyUrl);
        if (response.ok) {
          const html = await response.text();
          
          // Extract some basic article info from HTML
          const titles = html.match(/<h2[^>]*>(.*?)<\/h2>/g) || [];
          const scraped = titles.slice(0, 10).map((title, i) => {
            // Extract text from title
            const text = title.replace(/<[^>]*>/g, '');
            return {
              id: `scraped-${i}`,
              title: text.trim(),
              description: "Latest UFC news from ufc.com",
              url: "https://www.ufc.com/news",
              imageUrl: '/static/placeholder.png',
              source: "UFC.com",
              publishedAt: new Date().toISOString(),
              imageLoaded: true
            };
          });
          
          if (scraped.length > 0) {
            console.log(`Extracted ${scraped.length} articles from UFC.com`);
            return scraped;
          }
        }
      } catch (scrapeError) {
        console.error("Scraping fallback failed:", scrapeError);
      }
      
      // Return fallback content as absolute last resort
      return generateFallbackNews();
    }
  }
};

// Helper function to generate fallback news items when all sources fail
function generateFallbackNews() {
  // Create at least 5 static news items as fallback
  return [
    {
      id: "fallback-1",
      title: "UFC News Currently Unavailable",
      description: "We're working to restore the latest UFC news feed. Please check back later or visit UFC.com for the latest updates.",
      url: "https://www.ufc.com/news",
      imageUrl: "/static/placeholder.png",
      source: "UFC Predictor",
      publishedAt: new Date().toISOString(),
      imageLoaded: true
    },
    {
      id: "fallback-2",
      title: "Explore Fighter Statistics",
      description: "While we restore the news feed, explore fighter statistics and make your own fight predictions!",
      url: "/fighters",
      imageUrl: "/static/placeholder.png",
      source: "UFC Predictor",
      publishedAt: new Date().toISOString(),
      imageLoaded: true
    },
    {
      id: "fallback-3",
      title: "Try Our Fight Predictor",
      description: "Use our advanced machine learning model to predict the outcome of UFC matchups.",
      url: "/predict",
      imageUrl: "/static/placeholder.png",
      source: "UFC Predictor",
      publishedAt: new Date().toISOString(),
      imageLoaded: true
    },
    {
      id: "fallback-4",
      title: "Visit Official UFC Website",
      description: "For the latest official UFC news, event schedules, and fighter information.",
      url: "https://www.ufc.com",
      imageUrl: "/static/placeholder.png",
      source: "UFC Predictor",
      publishedAt: new Date().toISOString(),
      imageLoaded: true
    },
    {
      id: "fallback-5",
      title: "Check Back Soon",
      description: "Our live news feed will be back shortly with the latest UFC updates and fight news.",
      url: "https://www.ufc.com/news",
      imageUrl: "/static/placeholder.png",
      source: "UFC Predictor",
      publishedAt: new Date().toISOString(),
      imageLoaded: true
    }
  ];
}

export default api;
