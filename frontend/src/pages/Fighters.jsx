import React, { useState, useEffect } from 'react';
import { fighterService } from '../services/api';
import FighterCard from '../components/fighters/FighterCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

const Fighters = () => {
  const [fighters, setFighters] = useState([]);
  const [filteredFighters, setFilteredFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [weightClassFilter, setWeightClassFilter] = useState('All');

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

  useEffect(() => {
    // Filter fighters based on search term and weight class
    let results = fighters;
    
    if (searchTerm) {
      results = results.filter(fighter => 
        fighter.name && fighter.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    if (weightClassFilter !== 'All') {
      // This is an approximation as weight class might be calculated differently
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
    
    setFilteredFighters(results);
  }, [searchTerm, weightClassFilter, fighters]);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">UFC Fighters</h1>
      
      <div className="mb-8 flex flex-col md:flex-row gap-4">
        <div className="flex-grow">
          <input
            type="text"
            placeholder="Search fighters..."
            className="w-full p-2 bg-gray-800 border border-gray-700 rounded text-white"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div>
          <select
            className="p-2 bg-gray-800 border border-gray-700 rounded text-white"
            value={weightClassFilter}
            onChange={(e) => setWeightClassFilter(e.target.value)}
          >
            {weightClasses.map(weightClass => (
              <option key={weightClass} value={weightClass}>
                {weightClass}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {loading ? (
        <LoadingSpinner text="Loading fighters..." />
      ) : error ? (
        <ErrorAlert message={error} />
      ) : (
        <>
          <p className="mb-4 text-gray-400">
            {filteredFighters.length} fighters found
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredFighters.map(fighter => (
              <FighterCard key={fighter.id || fighter.name} fighter={fighter} />
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default Fighters;