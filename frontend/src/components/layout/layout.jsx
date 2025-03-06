import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import { AnimatePresence, motion } from 'framer-motion';

// This Layout component will ensure the proper spacing around your content
// and adds a consistent page transition effect
const Layout = () => {
  return (
    <div className="flex flex-col min-h-screen bg-gray-900">
      <Navbar />
      
      {/* Main content area with padding for fixed navbar */}
      <main className="flex-grow pt-20 px-4 md:px-6 container mx-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="py-6"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
      
      <Footer />
    </div>
  );
};

export default Layout;