import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import arenaBg from '@/assets/arena-bg.jpg';

interface GameLauncherProps {
  onGameSelect: (game: 'fighting' | 'badminton' | 'racing') => void;
}

const GameLauncher: React.FC<GameLauncherProps> = ({ onGameSelect }) => {
  interface Game {
    id: 'fighting' | 'badminton' | 'racing';
    title: string;
    description: string;
    color: string;
    bgColor: string;
    icon: string;
    gradient: string;
    buttonText: string;
  }

  const games: Game[] = [
    {
      id: 'fighting' as const,
      title: 'FIGHTING ARENA',
      description: 'Engage in intense combat with hyper-realistic physics and dynamic characters.',
      color: 'text-primary',
      bgColor: 'from-primary/30 to-gaming-teal/20',
      icon: '🥊',
      gradient: 'bg-gradient-to-br from-primary/20 via-gaming-teal/10 to-gaming-purple/15',
      buttonText: 'Enter Arena'
    },
    {
      id: 'badminton' as const,
      title: 'BADMINTON COURT',
      description: 'Experience pro-level badminton with precise shuttlecock physics and fluid racket controls.',
      color: 'text-gaming-purple',
      bgColor: 'from-gaming-purple/30 to-gaming-magenta/20',
      icon: '🏸',
      gradient: 'bg-gradient-to-br from-gaming-purple/20 via-accent/10 to-gaming-teal/15',
      buttonText: 'Start Match'
    },
    {
      id: 'racing' as const,
      title: 'RACING CIRCUIT',
      description: 'Push the limits with high-octane racing, featuring stunning tracks and realistic vehicle handling.',
      color: 'text-gaming-orange',
      bgColor: 'from-gaming-orange/30 to-gaming-yellow/20',
      icon: '🏎️',
      gradient: 'bg-gradient-to-br from-gaming-orange/20 via-gaming-yellow/10 to-primary/15',
      buttonText: 'Begin Race'
    }
  ];

  return (
    <div
      className="min-h-screen bg-cover bg-center bg-no-repeat flex items-center justify-center p-4 overflow-auto"
      style={{ backgroundImage: `url(${arenaBg})` }}
    >
      <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" />
      
      <div className="relative z-10 max-w-5xl w-full">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-8"
        >
          <div className="mb-2">
            <h1 className="text-4xl md:text-5xl font-gaming font-extrabold tracking-wide text-foreground text-glow">
              MULTI GAME ARENA
            </h1>
          </div>
          <div className="text-lg md:text-xl font-medium text-foreground/90">
            One destination, three electrifying challenges. Test your skills in fighting, badminton, and car racing.
          </div>
        </motion.div>

        {/* Game Selection Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {games.map((game, index) => (
            <motion.div
              key={game.id}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              whileHover={{ scale: 1.05, y: -10 }}
              whileTap={{ scale: 0.95 }}
            >
              <Card
                className={`cursor-pointer h-80 overflow-hidden relative group border-2 border-border/50 hover:border-primary/60 transition-all duration-500 backdrop-blur-sm ${game.gradient} hover:scale-[1.02] active:scale-[0.98]`}
                onClick={() => onGameSelect(game.id)}
              >
                <CardContent className="p-6 h-full flex flex-col justify-between relative z-10">
                  <div className="text-center">
                    <div className="text-5xl mb-4 group-hover:scale-110 transition-transform duration-300 filter drop-shadow-lg">
                      {game.icon}
                    </div>
                    <h3 className={`text-2xl font-gaming font-bold mb-3 ${game.color} tracking-wide`}>
                      {game.title}
                    </h3>
                    <p className="text-foreground/80 text-sm leading-relaxed">
                      {game.description}
                    </p>
                  </div>
                  
                  <motion.div
                    className="mt-6"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="btn-gaming w-full py-3 text-center font-gaming font-bold text-base tracking-widest">
                      {game.buttonText}
                    </div>
                  </motion.div>
                </CardContent>
                
                {/* Professional glow effect */}
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="absolute inset-0 bg-gradient-gaming opacity-0 group-hover:opacity-5 transition-opacity duration-500" />
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.8 }}
          className="text-center mt-8"
        >
          <p className="text-sm text-muted-foreground">
            Use WASD for movement • J/K for combat • Space for actions • Click and drag to rotate view
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default GameLauncher;
