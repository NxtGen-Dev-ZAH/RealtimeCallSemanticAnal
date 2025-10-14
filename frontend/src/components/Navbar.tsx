'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BarChart3, History, Info, Brain, Sparkles } from 'lucide-react';

const Navbar = () => {
  const pathname = usePathname();

  const navItems = [
    { path: '/', label: 'Home', icon: BarChart3 },
    { path: '/history', label: 'History', icon: History },
    { path: '/about', label: 'About', icon: Info },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-white/10 border-b border-white/20">
      <div className="container mx-auto px-6">
        <div className="flex justify-between items-center h-20">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl blur opacity-75"></div>
              <div className="relative bg-gradient-to-r from-purple-500 to-pink-500 p-2 rounded-xl">
                <Brain className="h-8 w-8 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">
                Call Analysis AI
              </h1>
              <p className="text-xs text-purple-200/70 flex items-center">
                <Sparkles className="h-3 w-3 mr-1" />
                Powered by Advanced ML
              </p>
            </div>
          </div>
          
          <div className="flex space-x-2">
            {navItems.map(({ path, label, icon: Icon }) => {
              const isActive = pathname === path;
              return (
                <Link
                  key={path}
                  href={path}
                  className={`group flex items-center space-x-2 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-300 ${
                    isActive
                      ? 'bg-white/20 text-white shadow-lg backdrop-blur-sm border border-white/30'
                      : 'text-white/80 hover:text-white hover:bg-white/10 hover:backdrop-blur-sm border border-transparent hover:border-white/20'
                  }`}
                >
                  <Icon className={`h-4 w-4 transition-transform duration-300 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`} />
                  <span>{label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
