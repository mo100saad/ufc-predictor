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

  // Define weight classes with proper kg ranges (using the same thresholds as in your FighterCard)
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

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Hero Section */}
      <div className="relative mb-12">
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
      
      {/* Search and Filter Controls */}
      <motion.div 
        className="mb-8 p-6 backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="flex flex-col md:flex-row gap-4 mb-4">
          {/* Search Input */}
          <div className="flex-grow">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
                </svg>
              </div>
              <input
                type="text"
                placeholder="Search fighters by name..."
                className="w-full pl-10 p-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
          
          {/* Filter Dropdown - Weight Class */}
          <div className="md:w-56">
            <div className="relative">
              <label className="block text-sm text-gray-400 mb-1 pl-1">Weight Class</label>
              <select
                className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg text-white appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={weightClassFilter}
                onChange={(e) => setWeightClassFilter(e.target.value)}
              >
                {weightClasses.map(weightClass => (
                  <option key={weightClass} value={weightClass}>
                    {weightClass} {weightClass !== 'All' ? `(${getWeightClassRange(weightClass)})` : ''}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none mt-6">
                <svg className="h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
          </div>
          
          {/* Filter Dropdown - Stance */}
          <div className="md:w-56">
            <div className="relative">
              <label className="block text-sm text-gray-400 mb-1 pl-1">Stance</label>
              <select
                className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg text-white appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={stanceFilter}
                onChange={(e) => setStanceFilter(e.target.value)}
              >
                {stanceOptions.map(stance => (
                  <option key={stance} value={stance}>
                    {stance}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none mt-6">
                <svg className="h-5 w-5 text-gray-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex flex-col md:flex-row justify-between items-end">
          {/* Sort Options */}
          <div className="md:w-64 mb-4 md:mb-0">
            <div className="relative">
              <label className="block text-sm text-gray-400 mb-1 pl-1">Sort By</label>
              <div className="flex">
                <select
                  className="w-full p-3 bg-gray-800 border border-gray-700 rounded-l-lg text-white appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                  className="px-3 bg-gray-700 border border-gray-600 rounded-r-lg focus:outline-none hover:bg-gray-600"
                  onClick={toggleSortDirection}
                  title={sortDirection === 'asc' ? 'Ascending' : 'Descending'}
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
          
          {/* View Toggle Button */}
          <button 
            className={`px-4 py-2 rounded-lg transition-colors flex items-center border ${
              statsView 
                ? 'bg-blue-600 border-blue-500 hover:bg-blue-700 text-white' 
                : 'bg-gray-800 border-gray-700 hover:bg-gray-700 text-gray-300'
            }`}
            onClick={toggleStatsView}
          >
            <svg className="h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path d="M5 3a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V5a2 2 0 00-2-2H5zM3 5a2 2 0 012-2h10a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V5z" />
            </svg>
            {statsView ? 'Card View' : 'Stats View'}
          </button>
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
          {/* Results Summary */}
          <motion.div 
            className="flex justify-between items-center mb-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <p className="text-gray-400">
              <span className="font-medium text-white">{filteredFighters.length}</span> fighters found
              {weightClassFilter !== 'All' && (
                <span> in <span className="text-blue-400">{weightClassFilter}</span> division</span>
              )}
              {stanceFilter !== 'All' && (
                <span> with <span className="text-purple-400">{stanceFilter}</span> stance</span>
              )}
              {searchTerm && (
                <span> matching <span className="text-green-400">{searchTerm}</span></span>
              )}
            </p>
            
            <div className="text-sm text-gray-500">
              Sorted by <span className="text-gray-300">{sortOptions.find(o => o.value === sortOption)?.label || 'Name'}</span> ({sortDirection === 'asc' ? 'ascending' : 'descending'})
            </div>
          </motion.div>
          
          {/* Fighter Display */}
          {filteredFighters.length > 0 ? (
            <>
              {statsView ? (
                // Stats Table View
                <motion.div 
                  className="overflow-x-auto"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-gray-800 text-left">
                        <th className="p-3 border-b border-gray-700">Name</th>
                        <th className="p-3 border-b border-gray-700">Record</th>
                        <th className="p-3 border-b border-gray-700">Weight (kg)</th>
                        <th className="p-3 border-b border-gray-700">Height</th>
                        <th className="p-3 border-b border-gray-700">Reach</th>
                        <th className="p-3 border-b border-gray-700">Stance</th>
                        <th className="p-3 border-b border-gray-700">Style</th>
                        <th className="p-3 border-b border-gray-700">SLpM</th>
                        <th className="p-3 border-b border-gray-700">TD Avg</th>
                        <th className="p-3 border-b border-gray-700">Sub Avg</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredFighters.slice(0, displayCount).map((fighter, index) => (
                        <tr key={fighter.id || index} className={index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800/50'}>
                          <td className="p-3 border-b border-gray-700 font-medium text-white">
                            <Link 
                              to={`/fighter/${encodeURIComponent(fighter.name)}`}
                              className="hover:text-blue-400 transition-colors"
                            >
                              {fighter.name}
                            </Link>
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.wins || 0}-{fighter.losses || 0}
                            {fighter.draws > 0 ? `-${fighter.draws}` : ''}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.weight ? fighter.weight.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.height ? fighter.height.toFixed(2) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.reach ? fighter.reach.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.stance || 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {getFighterStyle(fighter)}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.SLpM ? fighter.SLpM.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.td_avg ? fighter.td_avg.toFixed(1) : 'N/A'}
                          </td>
                          <td className="p-3 border-b border-gray-700">
                            {fighter.sub_avg ? fighter.sub_avg.toFixed(1) : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </motion.div>
              ) : (
                // Card Grid View
                <motion.div 
                  className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
                  variants={container}
                  initial="hidden"
                  animate="show"
                >
                  {filteredFighters.slice(0, displayCount).map((fighter, index) => (
                    <motion.div key={fighter.id || index} variants={item}>
                      <FighterCard fighter={fighter} />
                    </motion.div>
                  ))}
                </motion.div>
              )}
              
              {/* Load More Button */}
              {displayCount < filteredFighters.length && (
                <motion.div 
                  className="mt-10 flex justify-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  <button 
                    className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium rounded-lg shadow-lg transition-all duration-300 transform hover:scale-105"
                    onClick={loadMore}
                  >
                    Load More Fighters
                  </button>
                </motion.div>
              )}
            </>
          ) : (
            <motion.div 
              className="text-center py-16"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <p className="text-gray-400 text-lg mb-4">No fighters found matching your criteria</p>
              <button
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
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