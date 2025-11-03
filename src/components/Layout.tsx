import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '@/lib/auth';
import { Button } from '@/components/ui/button';
import { 
  Brain, 
  LayoutDashboard, 
  Zap, 
  History, 
  RefreshCw, 
  LogOut,
  ChevronRight
} from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const { org, signout } = useAuth();
  const location = useLocation();

  const navItems = [
    { path: '/dashboard/summary', label: 'Summary', icon: LayoutDashboard },
    { path: '/dashboard/predict', label: 'Predict', icon: Zap },
    { path: '/dashboard/history', label: 'History', icon: History },
    { path: '/dashboard/retrain', label: 'Retrain', icon: RefreshCw },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-lg sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Link to="/dashboard/summary" className="flex items-center gap-3 group">
              <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                <Brain className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">Grade Predictor</h1>
                {org && <p className="text-xs text-muted-foreground">{org.name}</p>}
              </div>
            </Link>

            <Button
              variant="ghost"
              onClick={signout}
              className="gap-2 text-muted-foreground hover:text-foreground"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-12 gap-6">
          {/* Sidebar */}
          <aside className="col-span-12 lg:col-span-3">
            <nav className="space-y-1 sticky top-24">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`
                      flex items-center gap-3 px-4 py-3 rounded-lg transition-all
                      ${isActive 
                        ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/20' 
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                      }
                    `}
                  >
                    <Icon className="h-5 w-5" />
                    <span className="font-medium">{item.label}</span>
                    {isActive && <ChevronRight className="h-4 w-4 ml-auto" />}
                  </Link>
                );
              })}
            </nav>
          </aside>

          {/* Main Content */}
          <main className="col-span-12 lg:col-span-9">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}
