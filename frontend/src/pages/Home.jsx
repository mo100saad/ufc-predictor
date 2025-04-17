import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { fighterService, newsService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import FighterCard from '../components/fighters/FighterCard';
import { motion } from 'framer-motion';

const Home = () => {
  // Move state variables to the main component
  const [topFighters, setTopFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newsItems, setNewsItems] = useState([]);
  const [newsLoading, setNewsLoading] = useState(true);
  
  // Ref for news slider container
  const newsSliderRef = useRef(null);
  
  // Calculate win percentage with minimum fight requirement
  const calculateWinPercentage = (fighter) => {
    const wins = fighter.wins || 0;
    const losses = fighter.losses || 0;
    const totalFights = wins + losses;
    
    // Require at least 5 fights for consideration
    if (totalFights < 18) return 0;
    return wins / totalFights;
  };
  
  // Load UFC news from NewsAPI through backend 
  const loadUFCNews = async () => {
    try {
      setNewsLoading(true);
      
      // Fetch news from our backend API
      const news = await newsService.getUFCNews();
      
      if (news && news.length > 0) {
        // Format dates and process news items
        const formattedNews = news.map(item => ({
          ...item,
          // Ensure date is properly formatted
          date: item.publishedAt || new Date().toISOString(),
          // Add initial image loading state
          imageLoaded: false
        }));
        
        setNewsItems(formattedNews);
      } else {
        // No news found
        setNewsItems([]);
      }
      
      setNewsLoading(false);
    } catch (err) {
      console.error("Failed to load UFC news:", err);
      setNewsLoading(false);
      // Don't set error state - we want the rest of the page to work even if news fails
      setNewsItems([]);
    }
  };

  useEffect(() => {
    const loadFighters = async () => {
      try {
        setLoading(true);
        const fighters = await fighterService.getAllFighters();
        
        // Add win percentage to each fighter
        const fightersWithWinPct = fighters.map(fighter => ({
          ...fighter,
          winPct: calculateWinPercentage(fighter)
        }))
        // Filter out fighters with < 80% win rate
        .filter(fighter => fighter.winPct >= 0.8)
        // Sort by win percentage (highest first)
        .sort((a, b) => b.winPct - a.winPct)
        // Take top 5
        .slice(0, 5);
        
        setTopFighters(fightersWithWinPct);
        setLoading(false);
      } catch (err) {
        console.error("Failed to load fighters:", err);
        setError("Failed to load fighters. Please try again later.");
        setLoading(false);
      }
    };
    
    // Load both fighters and news
    loadFighters();
    loadUFCNews();
  }, []);

  // Framer Motion animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900"
    >
      {/* Hero Section with Refined Design */}
      <div className="relative min-h-screen flex flex-col">
        {/* Navbar Placeholder - Add actual navbar here with pt-4 or pt-6 */}
        <div className="relative z-20 pt-6">
          {/* Your Navbar Component Would Go Here */}
        </div>

        {/* Background Overlay */}
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-20" 
          style={{ 
            backgroundImage: "url('/path-to-your-hero-image.jpg')",
            filter: 'grayscale(50%) brightness(30%)'
          }}
        ></div>

        {/* Content Container */}
        <div className="relative z-10 container mx-auto px-4 flex-grow flex flex-col">
          {/* Hero Content */}
          <div className="flex-grow flex flex-col justify-center">
            <div className="text-center">
              {/* Refined Title with Elegant Typography */}
              <h1 
                className="text-6xl md:text-8xl font-bebas-neue tracking-wide uppercase text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-red-600 to-red-700 mb-6 drop-shadow-lg"
                style={{ 
                  WebkitTextStroke: '1px rgba(255,255,255,0.2)', 
                }}
              >
                UFC Predictor
              </h1>

              {/* Subtitle with Clear Visibility */}
              <p className="text-lg md:text-xl max-w-2xl mx-auto text-gray-200 mb-10 leading-relaxed font-light opacity-100">
                Harness the power of advanced machine learning to predict UFC fight outcomes with precision. Analyze fighter statistics and get accurate predictions for upcoming matches.
              </p>

              {/* Action Buttons */}
              <motion.div 
                className="flex justify-center space-x-6 mb-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.5 }}
              >
                <Link 
                  to="/predict" 
                  className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white rounded-full transform hover:scale-105 transition duration-300 shadow-lg uppercase tracking-wider"
                >
                  Predict a Fight
                </Link>
                <Link 
                  to="/fighters" 
                  className="px-8 py-3 border-2 border-white text-white hover:bg-white hover:text-black rounded-full transform hover:scale-105 transition duration-300 shadow-lg uppercase tracking-wider"
                >
                  Browse Fighters
                </Link>
              </motion.div>
            </div>
          </div>

          {/* Content Grid - Restructured with three sections stacked */}
          <div className="flex flex-col space-y-8 mb-16">
            {/* UFC Latest News Section - New addition */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              <div className="flex justify-between items-center mb-6 border-b border-gray-700 pb-3">
                <motion.h2 
                  className="text-2xl font-bold text-white"
                  variants={itemVariants}
                >
                  UFC Latest News
                </motion.h2>
                <motion.a
                  href="https://www.ufc.com/trending/all"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-blue-400 hover:text-blue-300"
                  variants={itemVariants}
                >
                  View more news →
                </motion.a>
              </div>
              
              {newsLoading ? (
                <div className="flex justify-center py-6">
                  <LoadingSpinner text="Loading UFC news..." />
                </div>
              ) : newsItems.length === 0 ? (
                <div className="text-center text-gray-400 py-6">
                  No news available at the moment
                </div>
              ) : (
                <div className="relative">
                  {/* Horizontal scrolling container */}
                  <div 
                    ref={newsSliderRef} 
                    className="flex overflow-x-auto pb-4 -mx-2 px-2 scroll-smooth"
                    style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                  >
                    {newsItems.map((newsItem) => (
                      <motion.div
                        key={newsItem.id}
                        variants={itemVariants}
                        whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.2 } }}
                        className="flex-shrink-0 w-full sm:w-[350px] md:w-[400px] mx-2 bg-gradient-to-br from-gray-700/50 to-gray-800/60 rounded-lg overflow-hidden border border-gray-600/30 shadow-lg hover:shadow-xl transition-all duration-300"
                      >
                        <a href={newsItem.url} target="_blank" rel="noopener noreferrer" className="block h-full">
                          {/* Much larger image container */}
                          <div className="h-48 sm:h-56 bg-gray-700 overflow-hidden relative">
                            {/* Loading state for image */}
                            <div className="absolute inset-0 flex items-center justify-center bg-gray-800/70 z-10 transition-opacity duration-300" 
                                 style={{ opacity: newsItem.imageLoaded ? 0 : 1 }}>
                              <LoadingSpinner size="sm" />
                            </div>
                            <img 
                              src={newsItem.imageUrl} 
                              alt={newsItem.title}
                              onLoad={() => newsItem.imageLoaded = true}
                              onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = '/static/placeholder.png';
                                newsItem.imageLoaded = true;
                              }}
                              className="w-full h-full object-cover transition-all duration-500 transform hover:scale-105"
                              style={{ opacity: newsItem.imageLoaded ? 1 : 0.3 }}
                            />
                          </div>
                          
                          <div className="p-4">
                            <h3 className="text-white font-bold text-lg mb-2 line-clamp-2 hover:text-blue-400 transition-colors">
                              {newsItem.title}
                            </h3>
                            
                            <p className="text-gray-400 text-sm line-clamp-2 mb-2">
                              {newsItem.description}
                            </p>
                            
                            <div className="flex justify-between items-center mt-3 text-xs text-gray-500">
                              <span className="flex items-center flex-wrap">
                                <span className="mr-2">{new Date(newsItem.date || newsItem.publishedAt).toLocaleDateString()}</span>
                                {newsItem.source && (
                                  <span className="bg-gray-800 px-1.5 py-0.5 rounded text-[10px] text-gray-400">
                                    {newsItem.source}
                                  </span>
                                )}
                              </span>
                              <span className="text-blue-400 hover:underline">Read more →</span>
                            </div>
                          </div>
                        </a>
                      </motion.div>
                    ))}
                  </div>

                  {/* Left/Right scroll indicators with actual functionality */}
                  <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10 block">
                    <button 
                      className="bg-gray-800/90 hover:bg-gray-700 text-white p-2 rounded-full shadow-lg hover:scale-110 transition-all duration-200"
                      onClick={() => {
                        if (newsSliderRef.current) {
                          newsSliderRef.current.scrollBy({ left: -400, behavior: 'smooth' });
                        }
                      }}
                      aria-label="Scroll left"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                      </svg>
                    </button>
                  </div>
                  <div className="absolute right-2 top-1/2 transform -translate-y-1/2 z-10 block">
                    <button 
                      className="bg-gray-800/90 hover:bg-gray-700 text-white p-2 rounded-full shadow-lg hover:scale-110 transition-all duration-200"
                      onClick={() => {
                        if (newsSliderRef.current) {
                          newsSliderRef.current.scrollBy({ left: 400, behavior: 'smooth' });
                        }
                      }}
                      aria-label="Scroll right"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
            
            {/* Top Fighters Section - Full width */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              variants={containerVariants}
              initial="hidden"
              animate="show"
            >
              <motion.h2 
                className="text-2xl font-bold mb-6 text-white border-b border-gray-700 pb-3 text-center"
                variants={itemVariants}
              >
                Top Fighters
              </motion.h2>

              {loading ? (
                <LoadingSpinner text="Loading fighters..." />
              ) : error ? (
                <ErrorAlert message={error} />
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  {topFighters.map((fighter, index) => (
                    <motion.div 
                      key={index}
                      variants={itemVariants}
                      whileHover={{ y: -5, transition: { duration: 0.2 } }}
                    >
                      {/* Using the same FighterCard component for consistency */}
                      <FighterCard fighter={fighter} />
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>

            {/* Statistics Section - Full width */}
            <motion.div 
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-6 shadow-2xl backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.5 }}
            >
              <h2 className="text-2xl font-bold mb-6 text-white border-b border-gray-700 pb-3 text-center">
                Prediction Statistics
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-700/50 rounded-lg p-6 text-center flex flex-col justify-center items-center border border-gray-600/30 shadow-md h-[120px]">
                  <div className="text-4xl font-bold text-red-500 mb-2">70-90%</div>
                  <div className="text-sm text-gray-300">
                    Prediction Accuracy for Championship Fights
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-6 text-center flex flex-col justify-center items-center border border-gray-600/30 shadow-md h-[120px]">
                  <div className="text-4xl font-bold text-blue-500 mb-2">2,450+</div>
                  <div className="text-sm text-gray-300">
                    UFC Fighters in Database
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded-lg p-6 text-center flex flex-col justify-center items-center border border-gray-600/30 shadow-md h-[120px]">
                  <div className="text-4xl font-bold text-purple-500 mb-2">12+</div>
                  <div className="text-sm text-gray-300">
                    Fight Statistics Analyzed Per Prediction
                  </div>
                </div>
              </div>
            </motion.div>
            
            {/* GitHub Button - Increased Size and Visibility */}
            <motion.div
              className="mt-8 flex justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9, duration: 0.5 }}
            >
              <a 
                href="https://github.com/mo100saad/ufc-predictor" 
                target="_blank" 
                rel="noopener noreferrer"
                className="group flex items-center px-8 py-4 bg-gray-800 hover:bg-gray-700 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 border border-gray-700 hover:border-gray-600"
              >
                <svg className="h-8 w-8 mr-3 text-white" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
                <div>
                  <span className="text-lg font-bold block">View on GitHub</span>
                  <span className="text-gray-400 text-sm">Star this project if you find it useful!</span>
                </div>
                <svg className="h-6 w-6 ml-2 opacity-0 group-hover:opacity-100 transform translate-x-0 group-hover:translate-x-1 transition-all duration-300" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </a>
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Home;