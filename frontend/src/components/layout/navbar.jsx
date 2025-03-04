import React from 'react';
import { Link, NavLink } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="bg-gray-900 shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-4">
          <Link to="/" className="flex items-center">
            <span className="text-2xl font-bold text-ufc-red">UFC</span>
            <span className="text-2xl font-bold ml-2 text-white">Predictor</span>
          </Link>
          
          <div className="flex space-x-6">
            <NavLink 
              to="/" 
              className={({ isActive }) => 
                isActive ? "font-bold text-ufc-red" : "text-gray-300 hover:text-ufc-red transition"
              }
              end
            >
              Home
            </NavLink>
            <NavLink 
              to="/fighters" 
              className={({ isActive }) => 
                isActive ? "font-bold text-ufc-red" : "text-gray-300 hover:text-ufc-red transition"
              }
            >
              Fighters
            </NavLink>
            <NavLink 
              to="/predict" 
              className={({ isActive }) => 
                isActive ? "font-bold text-ufc-red" : "text-gray-300 hover:text-ufc-red transition"
              }
            >
              Predict
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;