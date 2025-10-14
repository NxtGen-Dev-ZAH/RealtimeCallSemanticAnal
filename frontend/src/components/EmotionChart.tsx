'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface EmotionChartProps {
  data: Record<string, number>;
}

const EmotionChart = ({ data }: EmotionChartProps) => {
  // Transform data for the chart
  const chartData = Object.entries(data).map(([emotion, value]) => ({
    name: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: value,
    percentage: ((value / Object.values(data).reduce((a, b) => a + b, 0)) * 100).toFixed(1),
  }));

  // Color palette for emotions
  const COLORS = {
    happy: '#22c55e',
    neutral: '#6b7280',
    sad: '#3b82f6',
    angry: '#ef4444',
    fearful: '#f59e0b',
    surprised: '#8b5cf6',
    disgusted: '#10b981',
  };

  const getColor = (emotion: string) => {
    return COLORS[emotion.toLowerCase() as keyof typeof COLORS] || '#6b7280';
  };

  const formatTooltip = (value: number, name: string, props: any) => {
    const percentage = props.payload.percentage;
    return [`${value.toFixed(2)} (${percentage}%)`, name];
  };

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percentage }) => `${name} (${percentage}%)`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry.name)} />
            ))}
          </Pie>
          <Tooltip formatter={formatTooltip} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EmotionChart;
