import {
  Activity, BarChart3, Camera, Flame, LayoutDashboard,
  ScanLine, Settings2
} from 'lucide-react'

export type PageName = 'Overview' | 'AI Models' | 'Image Analysis' | 'Results' | 'Model Comparison' | 'About'

const navigation = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'AI Models', icon: ScanLine },
  { label: 'Image Analysis', icon: Camera },
  { label: 'Results', icon: BarChart3 },
  { label: 'Model Comparison', icon: ScanLine },
] as const

type SidebarProps = {
  activePage: PageName
  onNavigate: (page: PageName) => void
}

export default function Sidebar({ activePage, onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Flame size={22} /></div>
        <div>
          <div className="brand-name">FireSight<span>AI</span></div>
          <div className="brand-caption">VISION INTELLIGENCE</div>
        </div>
      </div>

      <div className="nav-label">WORKSPACE</div>
      <nav className="nav-list">
        {navigation.map(({ label, icon: Icon }) => (
          <button
            key={label}
            className={`nav-item ${activePage === label ? 'active' : ''}`}
            onClick={() => onNavigate(label)}
          >
            <Icon size={18} strokeWidth={1.8} />
            <span>{label}</span>
            {activePage === label && <span className="nav-indicator" />}
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="sidebar-note">
          <Activity size={17} />
          <div>
            <strong>Computer Vision</strong>
            <span>Fire & smoke segmentation</span>
          </div>
        </div>
        <button
          className={`nav-item ${activePage === 'About' ? 'active' : ''}`}
          onClick={() => onNavigate('About')}
        >
          <Settings2 size={18} />
          <span>About project</span>
        </button>

      </div>
    </aside>
  )
}

