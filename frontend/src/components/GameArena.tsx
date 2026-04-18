{(['fighting', 'badminton', 'racing'] as const).map((game) => {
  const isLocked = game !== 'fighting';
  const isActive = gameType === game;
  return (
    <motion.button
      key={game}
      onClick={() => !isLocked && onGameChange(game)}
      className={`relative px-4 py-2 rounded-lg font-bold uppercase tracking-wider text-sm transition-all duration-300 ${
        isActive
          ? 'btn-gaming'
          : isLocked
          ? 'opacity-40 cursor-not-allowed border border-white/10 text-gray-500'
          : 'btn-gaming-outline'
      }`}
      whileHover={{ scale: isLocked ? 1 : 1.05 }}
      whileTap={{ scale: isLocked ? 1 : 0.95 }}
      title={isLocked ? 'Coming Soon' : game}
    >
      {game}
      {isLocked && (
        <span className="ml-1 text-[9px] tracking-widest text-gray-500">
          🔒
        </span>
      )}
    </motion.button>
  );
})}