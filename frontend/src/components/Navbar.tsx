'use client';

import Link from 'next/link';
import { BarChart3, History, Info, Brain } from 'lucide-react';

const navItems = [
  { href: '#home', label: 'Home', icon: BarChart3 },
  { href: '#history', label: 'History', icon: History },
  { href: '#results', label: 'Results', icon: BarChart3 },
  { href: '#about', label: 'About', icon: Info },
];

const Navbar = () => {

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-white/10 border-b border-white/20">
      <div className="container mx-auto px-6">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-900 to-black rounded-xl blur opacity-75"></div>
              <div className="relative bg-gradient-to-r from-blue-900 to-slate-950 p-2 rounded-xl">
                <Brain className="h-6 w-6 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
                Call Analysis AI
              </h1>
            </div>
          </div>
          
          <div className="flex space-x-2">
            {navItems.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                scroll
                className="group flex items-center space-x-2 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-300 text-white/80 hover:text-white hover:bg-white/10 hover:backdrop-blur-sm border border-transparent hover:border-white/20"
              >
                <Icon className="h-4 w-4 transition-transform duration-300 group-hover:scale-110" />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
