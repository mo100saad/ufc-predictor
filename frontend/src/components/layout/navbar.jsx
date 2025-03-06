import React, { useState, useEffect } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  // Add scroll detection for navbar appearance change
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <motion.nav 
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled 
          ? 'bg-gray-900/95 backdrop-blur-md shadow-lg py-2' 
          : 'bg-gradient-to-b from-black via-gray-900/90 to-transparent py-4'
      }`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center">
          <Link to="/" className="flex items-center group">
            <motion.div
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.2 }}
            >
              <span className="text-2xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-red-500 to-red-700 group-hover:from-red-400 group-hover:to-red-600 transition-all duration-300">UFC</span>
              <span className="text-2xl font-extrabold ml-2 text-white">Predictor</span>
            </motion.div>
          </Link>
          
          <div className="flex space-x-8">
            {[
              { path: '/', label: 'Home', exact: true },
              { path: '/fighters', label: 'Fighters', exact: false },
              { path: '/predict', label: 'Predict', exact: false }
            ].map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.exact}
                className={({ isActive }) => `relative font-medium ${
                  isActive ? 'text-white' : 'text-gray-300 hover:text-white'
                }`}
              >
                {({ isActive }) => (
                  <>
                    {item.label}
                    {isActive && (
                      <motion.div 
                        className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-red-500 to-blue-500 rounded-full"
                        layoutId="navbar-indicator"
                        transition={{ type: 'spring', duration: 0.5 }}
                      />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </div>
          
          {/* Mobile-friendly menu button (optional) */}
          <div className="md:hidden">
            <button className="text-white focus:outline-none">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar;