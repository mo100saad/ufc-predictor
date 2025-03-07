import React, { useState, useEffect } from 'react';
import { fighterService } from '../services/api';
import FighterCard from '../components/fighters/FighterCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const Fighters = () => {
  const [fighters, setFighters] = useState([]);
  const [filteredFighters, setFilteredFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [weightClassFilter, setWeightClassFilter] = useState('All');
  const [stanceFilter, setStanceFilter] = useState('All');
  const [sortOption, setSortOption] = useState('name');
  const [sortDirection, setSortDirection] = useState('asc');
  const [displayCount, setDisplayCount] = useState(24);
  const [statsView, setStatsView] = useState(false);

  // Define weight classes with proper kg ranges
  const weightClasses = [
    'All',
    'Flyweight',     // ≤ 56.8 kg (125 lbs)
    'Bantamweight',  // 56.8 - 61.3 kg (135 lbs)
    'Featherweight', // 61.3 - 65.9 kg (145 lbs)
    'Lightweight',   // 65.9 - 70.32 kg (155 lbs)
    'Welterweight',  // 70.32 - 77.2 kg (170 lbs)
    'Middleweight',  // 77.2 - 84.0 kg (185 lbs)
    'Light Heavyweight', // 84.0 - 93.1 kg (205 lbs)
    'Heavyweight'    // > 93.1 kg (> 205 lbs)
  ];

  // Define stance options based on database
  const stanceOptions = [
    'All',
    'Orthodox',
    'Southpaw',
    'Switch',
    'Open Stance'
  ];

  const sortOptions = [
    { value: 'name', label: 'Name' },
    { value: 'wins', label: 'Wins' },
    { value: 'winRate', label: 'Win %' },
    { value: 'SLpM', label: 'Striking' },
    { value: 'td_avg', label: 'Takedowns' },
    { value: 'sub_avg', label: 'Submissions' },
    { value: 'weight', label: 'Weight' },
    { value: 'height', label: 'Height' },
    { value: 'reach', label: 'Reach' }
  ];

  useEffect(() => {
    const loadFighters = async () => {
      try {
        setLoading(true);
        const data = await fighterService.getAllFighters();
        setFighters(data);
        setFilteredFighters(data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to load fighters:", err);
        setError("Failed to load fighters. Please try again later.");
        setLoading(false);
      }
    };

    loadFighters();
  }, []);

  // Filter and sort fighters
  useEffect(() => {
    // Start with all fighters
    let results = [...fighters];
    
    // Filter by search term
    if (searchTerm) {
      results = results.filter(fighter => 
        fighter.name && fighter.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Filter by weight class - using kg values matching the FighterCard thresholds
    if (weightClassFilter !== 'All') {
      results = results.filter(fighter => {
        // Determine weight class based on weight in kg
        const weight = fighter.weight || 0;
        
        switch(weightClassFilter) {
          case 'Flyweight': return weight <= 56.8;
          case 'Bantamweight': return weight > 56.8 && weight <= 61.3;
          case 'Featherweight': return weight > 61.3 && weight <= 65.9;
          case 'Lightweight': return weight > 65.9 && weight <= 70.32;
          case 'Welterweight': return weight > 70.32 && weight <= 77.2;
          case 'Middleweight': return weight > 77.2 && weight <= 84.0;
          case 'Light Heavyweight': return weight > 84.0 && weight <= 93.1;
          case 'Heavyweight': return weight > 93.1;
          default: return true;
        }
      });
    }
    
    // Filter by stance
    if (stanceFilter !== 'All') {
      results = results.filter(fighter => 
        fighter.stance && fighter.stance.toLowerCase() === stanceFilter.toLowerCase()
      );
    }
    
    // Sort fighters
    results.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortOption) {
        case 'name':
          aValue = a.name || '';
          bValue = b.name || '';
          return sortDirection === 'asc' 
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
        
        case 'wins':
          aValue = a.wins || 0;
          bValue = b.wins || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'winRate':
          const aTotal = (a.wins || 0) + (a.losses || 0);
          const bTotal = (b.wins || 0) + (b.losses || 0);
          aValue = aTotal > 0 ? (a.wins || 0) / aTotal * 100 : 0;
          bValue = bTotal > 0 ? (b.wins || 0) / bTotal * 100 : 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'SLpM':
          aValue = a.SLpM || 0;
          bValue = b.SLpM || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'td_avg':
          aValue = a.td_avg || 0;
          bValue = b.td_avg || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
        
        case 'sub_avg':
          aValue = a.sub_avg || 0;
          bValue = b.sub_avg || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'weight':
          aValue = a.weight || 0;
          bValue = b.weight || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'height':
          aValue = a.height || 0;
          bValue = b.height || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'reach':
          aValue = a.reach || 0;
          bValue = b.reach || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        default:
          return 0;
      }
    });
    
    setFilteredFighters(results);
  }, [searchTerm, weightClassFilter, stanceFilter, sortOption, sortDirection, fighters]);

  const toggleSortDirection = () => {
    setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
  };

  const loadMore = () => {
    setDisplayCount(prev => prev + 24);
  };

  const toggleStatsView = () => {
    setStatsView(prev => !prev);
  };

  // Convert weight class to display range in kg
  const getWeightClassRange = (weightClass) => {
    switch (weightClass) {
      case 'Flyweight': return '≤ 56.8 kg';
      case 'Bantamweight': return '56.8 - 61.3 kg';
      case 'Featherweight': return '61.3 - 65.9 kg';
      case 'Lightweight': return '65.9 - 70.32 kg';
      case 'Welterweight': return '70.32 - 77.2 kg';
      case 'Middleweight': return '77.2 - 84.0 kg';
      case 'Light Heavyweight': return '84.0 - 93.1 kg';
      case 'Heavyweight': return '> 93.1 kg';
      default: return '';
    }
  };
  
  // Helper to determine fighter style based on stats
  const getFighterStyle = (fighter) => {
    const strikingRatio = (fighter.SLpM || 0) / 5; // Normalized to 0-1 where 5 is high
    const grapplingRatio = ((fighter.td_avg || 0) + (fighter.sub_avg || 0)) / 6; // Normalized
    
    if (strikingRatio > 0.7 && grapplingRatio < 0.3) return "Striker";
    if (strikingRatio < 0.3 && grapplingRatio > 0.7) return "Grappler";
    if (strikingRatio > 0.5 && grapplingRatio > 0.5) return "Well-Rounded";
    return "Balanced";
  };

  // Animation variants
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.3 } }
  };
  
  // Create keyframes animation for the glow effect on the toggle button
  const pulseKeyframes = `
    @keyframes pulse-slow {
      0% { opacity: 0.2; }
      50% { opacity: 0.3; }
      100% { opacity: 0.2; }
    }
  `;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="max-w-screen-2xl mx-auto px-4 pt-6 pb-12"
    >
      {/* Custom keyframes for animations */}
      <style>{pulseKeyframes}</style>

      {/* Hero Section */}
      <div className="relative mb-8">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-gray-900 to-transparent h-48 -z-10"></div>
        
        <motion.h1 
          className="text-4xl font-extrabold pt-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-red-500 via-white to-blue-500"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          UFC Fighters Database
        </motion.h1>
        
        <motion.p 
          className="text-lg text-gray-300 max-w-2xl mx-auto text-center font-light mt-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.7 }}
        >
          Browse and search through our comprehensive database of UFC fighters
        </motion.p>
      </div>
      
     {/* Search and Filter Controls - Redesigned for better alignment and visual appeal */}
<motion.div 
  className="mb-6 rounded-xl overflow-hidden shadow-2xl"
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.5, delay: 0.2 }}
>
  {/* Top Search Bar - Full width with gradient background */}
  <div className="bg-gradient-to-r from-gray-900 to-gray-800 p-4">
    <div className="relative">
      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
        <svg className="h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
        </svg>
      </div>
      <input
        type="text"
        placeholder="Search fighters by name..."
        className="w-full pl-12 p-3.5 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
      />
    </div>
  </div>
  
  {/* Filter Controls Bar - Flex container with improved alignment */}
  <div className="bg-gray-900 p-4 flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0 md:space-x-4">
    {/* Left Side - Filters Group */}
    <div className="flex flex-col sm:flex-row items-start sm:items-center sm:justify-start space-y-4 sm:space-y-0 sm:space-x-8 w-full md:w-auto">
      {/* Weight Class Filter */}
      <div className="w-full sm:w-auto">
        <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
          <label className="text-sm text-gray-400 whitespace-nowrap font-medium">Weight Class:</label>
          <div className="relative flex-grow">
            <select
              className="w-full p-2.5 bg-gray-800 border border-gray-700 rounded-lg text-white appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all min-w-[200px]"
              value={weightClassFilter}
              onChange={(e) => setWeightClassFilter(e.target.value)}
            >
              {weightClasses.map(weightClass => (
                <option key={weightClass} value={weightClass}>
                  {weightClass !== 'All' ? weightClass : 'All Weight Classes'}
                </option>
              ))}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
              <svg className="h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path
                  fillRule="evenodd"
                  d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Stance Filter */}
      <div className="w-full sm:w-auto">
        <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
          <label className="text-sm text-gray-400 whitespace-nowrap font-medium">Stance:</label>
          <div className="relative flex-grow">
            <select
              className="w-full p-2.5 bg-gray-800 border border-gray-700 rounded-lg text-white appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all min-w-[140px]"
              value={stanceFilter}
              onChange={(e) => setStanceFilter(e.target.value)}
            >
              {stanceOptions.map(stance => (
                <option key={stance} value={stance}>
                  {stance !== 'All' ? stance : 'All Stances'}
                </option>
              ))}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
              <svg className="h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>

    {/* Right Side - Sort Controls and View Toggle */}
    <div className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-4 w-full md:w-auto">
      {/* Sort Controls */}
      <div className="w-full sm:w-auto">
        <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
          <label className="text-sm text-gray-400 whitespace-nowrap font-medium">Sort By:</label>
          <div className="relative flex-grow">
            <div className="flex">
              <select
                className="p-2.5 bg-gray-800 border border-gray-700 rounded-l-lg text-white appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 pl-3 pr-8 min-w-[120px] transition-all"
                value={sortOption}
                onChange={(e) => setSortOption(e.target.value)}
              >
                {sortOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button
                className="p-2.5 bg-gray-700 border border-gray-600 rounded-r-lg focus:outline-none hover:bg-gray-600 transition-all duration-300 transform hover:scale-105"
                onClick={toggleSortDirection}
                title={sortDirection === 'asc' ? 'Switch to Descending' : 'Switch to Ascending'}
              >
                {sortDirection === 'asc' ? (
                  <svg className="h-5 w-5 text-gray-300" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="h-5 w-5 text-gray-300" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

            {/* Enhanced View Toggle Button */}
            <button 
              className={`
                px-5 py-2.5 rounded-lg font-medium
                min-w-[140px] h-[42px]
                shadow-lg hover:shadow-xl
                transition-all duration-300 transform hover:translate-y-[-2px]
                flex items-center justify-center gap-2 overflow-hidden relative
                ${
                  statsView 
                    ? 'bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white' 
                    : 'bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800 text-gray-200 border border-gray-700'
                }
              `}
              onClick={toggleStatsView}
            >
              {/* Button Background Animation */}
              <div className={`
                absolute inset-0 bg-gradient-to-r 
                ${statsView 
                  ? 'from-blue-700 via-blue-500 to-blue-700' 
                  : 'from-gray-700 via-gray-600 to-gray-700'
                }
                opacity-0 hover:opacity-100 transition-opacity duration-500
              `}></div>
              
              {/* Icons with Fade Transition */}
              <div className="relative flex items-center justify-center w-full">
                {/* Card View Icon - Shown when in Stats View */}
                <div className={`
                  absolute inset-0 flex items-center justify-center
                  transition-all duration-300 transform
                  ${statsView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
                `}>
                  <svg className="h-5 w-5 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z" />
                  </svg>
                  <span>Card View</span>
                </div>
                
                {/* Stats View Icon - Shown when in Card View */}
                <div className={`
                  absolute inset-0 flex items-center justify-center
                  transition-all duration-300 transform
                  ${!statsView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
                `}>
                  <svg className="h-5 w-5 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zm6-4a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zm6-3a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                  </svg>
                  <span>Stats View</span>
                </div>
              </div>
              
              {/* Button Glow Effect */}
              <div className={`
                absolute inset-0 rounded-lg
                transition-opacity duration-500
                ${statsView 
                  ? 'opacity-30 animate-pulse-slow bg-blue-400' 
                  : 'opacity-0'
                }
                blur-xl
              `}></div>
            </button>
          </div>
        </div>
      </motion.div>
      
      {/* Results Summary Bar - Clean, aligned with the main controls */}
      <motion.div 
        className="flex flex-wrap justify-between items-center mb-5 px-4 py-3 bg-gray-900/60 backdrop-blur-sm rounded-lg border border-gray-800 shadow-md"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div className="text-gray-400">
          <span className="font-medium text-white">{filteredFighters.length}</span> fighters found
          {weightClassFilter !== 'All' && (
            <span> in <span className="text-blue-400">{weightClassFilter}</span> division</span>
          )}
          {stanceFilter !== 'All' && (
            <span> with <span className="text-purple-400">{stanceFilter}</span> stance</span>
          )}
          {searchTerm && (
            <span> matching <span className="text-green-400">"{searchTerm}"</span></span>
          )}
        </div>
        
        <div className="text-sm text-gray-500 mt-2 sm:mt-0">
          Sorted by <span className="text-gray-300">{sortOptions.find(o => o.value === sortOption)?.label || 'Name'}</span> ({sortDirection === 'asc' ? 'ascending' : 'descending'})
        </div>
      </motion.div>
      
      {/* Results Section */}
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner text="Loading fighters..." />
        </div>
      ) : error ? (
        <ErrorAlert message={error} />
      ) : (
        <>
          {/* Fighter Display */}
          {filteredFighters.length > 0 ? (
            <>
              {statsView ? (
                // Stats Table View
                <motion.div 
                  className="overflow-x-auto bg-gray-900/50 backdrop-blur-sm border border-gray-800 rounded-xl shadow-xl"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-gradient-to-r from-gray-800 to-gray-700 text-left">
                        <th className="p-3 border-b border-gray-700 font-semibold">Name</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Record</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Weight (kg)</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Height</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Reach</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Stance</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Style</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">SLpM</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">TD Avg</th>
                        <th className="p-3 border-b border-gray-700 font-semibold">Sub Avg</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredFighters.slice(0, displayCount).map((fighter, index) => (
                        <tr key={fighter.id || index} className={index % 2 === 0 ? 'bg-gray-900/80' : 'bg-gray-800/50'}>
                          <td className="p-3 border-b border-gray-700 font-medium text-white">
                            <Link 
                              to={`/fighters/${encodeURIComponent(fighter.name)}`}
                              className="hover:text-blue-400 transition-colors hover:underline flex items-center"
                            >
                              {fighter.name}
                              <svg className="h-4 w-4 ml-1 opacity-50" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                              </svg>
                            </Link>
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            <span className="py-1 px-2 rounded bg-gray-800 text-gray-300">
                              {fighter.wins || 0}-{fighter.losses || 0}
                              {fighter.draws > 0 ? `-${fighter.draws}` : ''}
                            </span>
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.weight !== null && fighter.weight !== undefined ? fighter.weight.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.height !== null && fighter.height !== undefined ? fighter.height.toFixed(2) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.reach !== null && fighter.reach !== undefined ? fighter.reach.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.stance !== null && fighter.stance !== undefined ? fighter.stance : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            <span className={`py-1 px-2 rounded text-sm ${
                              getFighterStyle(fighter) === 'Striker' ? 'bg-red-900/30 text-red-300' :
                              getFighterStyle(fighter) === 'Grappler' ? 'bg-blue-900/30 text-blue-300' :
                              getFighterStyle(fighter) === 'Well-Rounded' ? 'bg-purple-900/30 text-purple-300' :
                              'bg-gray-800 text-gray-300'
                            }`}>
                              {getFighterStyle(fighter)}
                            </span>
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.SLpM !== null && fighter.SLpM !== undefined ? fighter.SLpM.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.td_avg !== null && fighter.td_avg !== undefined ? fighter.td_avg.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.sub_avg !== null && fighter.sub_avg !== undefined ? fighter.sub_avg.toFixed(1) : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </motion.div>
              ) : (
                // Card Grid View with improved animations
                <motion.div 
                  className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
                  variants={container}
                  initial="hidden"
                  animate="show"
                >
                  {filteredFighters.slice(0, displayCount).map((fighter, index) => (
                    <motion.div 
                      key={fighter.id || index} 
                      variants={item}
                      whileHover={{ y: -5, transition: { duration: 0.2 } }}
                    >
                      <FighterCard fighter={fighter} />
                    </motion.div>
                  ))}
                </motion.div>
              )}
              
              {/* Load More Button - Enhanced */}
              {displayCount < filteredFighters.length && (
                <motion.div 
                  className="mt-10 flex justify-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  <button 
                    className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium rounded-lg shadow-lg transition-all duration-300 transform hover:scale-105 hover:shadow-xl"
                    onClick={loadMore}
                  >
                    <div className="flex items-center">
                      <span>Load More Fighters</span>
                      <svg className="ml-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clipRule="evenodd" />
                      </svg>
                    </div>
                  </button>
                </motion.div>
              )}
            </>
          ) : (
            <motion.div 
              className="text-center py-16 bg-gray-900/50 backdrop-blur-sm border border-gray-800 rounded-xl shadow-xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <svg className="mx-auto h-16 w-16 text-gray-600 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-gray-400 text-lg mb-4">No fighters found matching your criteria</p>
              <button
                className="px-5 py-2.5 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                onClick={() => {
                  setSearchTerm('');
                  setWeightClassFilter('All');
                  setStanceFilter('All');
                }}
              >
                Reset Filters
              </button>
            </motion.div>
          )}
        </>
      )}
    </motion.div>
  );
};

export default Fighters;