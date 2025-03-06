import React, { useState, useEffect } from 'react';
import { fighterService } from '../services/api';
import FighterCard from '../components/fighters/FighterCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { motion } from 'framer-motion';

const Fighters = () => {
  const [fighters, setFighters] = useState([]);
  const [filteredFighters, setFilteredFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [weightClassFilter, setWeightClassFilter] = useState('All');
  const [sortOption, setSortOption] = useState('name');
  const [sortDirection, setSortDirection] = useState('asc');
  const [displayCount, setDisplayCount] = useState(24);

  const weightClasses = [
    'All',
    'Flyweight',
    'Bantamweight',
    'Featherweight',
    'Lightweight',
    'Welterweight',
    'Middleweight',
    'Light Heavyweight',
    'Heavyweight'
  ];

  const sortOptions = [
    { value: 'name', label: 'Name' },
    { value: 'wins', label: 'Wins' },
    { value: 'winRate', label: 'Win %' },
    { value: 'SLpM', label: 'Striking' },
    { value: 'td_avg', label: 'Takedowns' }
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
    
    // Filter by weight class
    if (weightClassFilter !== 'All') {
      results = results.filter(fighter => {
        // Determine weight class based on weight
        const weight = fighter.weight || 0;
        
        switch(weightClassFilter) {
          case 'Flyweight': return weight <= 125;
          case 'Bantamweight': return weight > 125 && weight <= 135;
          case 'Featherweight': return weight > 135 && weight <= 145;
          case 'Lightweight': return weight > 145 && weight <= 155;
          case 'Welterweight': return weight > 155 && weight <= 170;
          case 'Middleweight': return weight > 170 && weight <= 185;
          case 'Light Heavyweight': return weight > 185 && weight <= 205;
          case 'Heavyweight': return weight > 205;
          default: return true;
        }
      });
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
          aValue = aTotal > 0 ? (a.wins || 0) / aTotal : 0;
          bValue = bTotal > 0 ? (b.wins || 0) / bTotal : 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'SLpM':
          aValue = a.SLpM || 0;
          bValue = b.SLpM || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        case 'td_avg':
          aValue = a.td_avg || 0;
          bValue = b.td_avg || 0;
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
          
        default:
          return 0;
      }
    });
    
    setFilteredFighters(results);
  }, [searchTerm, weightClassFilter, sortOption, sortDirection, fighters]);

  const toggleSortDirection = () => {
    setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
  };

  const loadMore = () => {
    setDisplayCount(prev => prev + 24);
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
        <div className="flex flex-col md:flex-row gap-4">
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
          
          {/* Filter Dropdown */}
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
                    {weightClass}
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
          
          {/* Sort Options */}
          <div className="md:w-56">
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
              {searchTerm && (
                <span> matching <span className="text-green-400">{searchTerm}</span></span>
              )}
            </p>
            
            <div className="text-sm text-gray-500">
              Sorted by <span className="text-gray-300">{sortOptions.find(o => o.value === sortOption)?.label || 'Name'}</span> ({sortDirection === 'asc' ? 'ascending' : 'descending'})
            </div>
          </motion.div>
          
          {/* Fighter Cards Grid */}
          {filteredFighters.length > 0 ? (
            <>
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