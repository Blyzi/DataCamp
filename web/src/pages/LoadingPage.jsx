import Logo from '../assets/logo.svg'

export default function LoadingPage({loading}) {
    

    return (
        <div className="h-screen w-screen bg-cyan-400 flex justify-center items-center">
            <div className='flex justify-center items-center flex-col gap-4'>
                <img src={Logo} alt="" className="fill-white h-24 w-24" />
                <div className='text-6xl text-white font-bold font-logo'>
                    TruthEyes
                </div>
                
                <div className="w-full rounded-full h-2.5 bg-cyan-700">
                    <div className="bg-white h-2.5 rounded-full" style={{'width': loading + '%'}}></div>
                </div>

            </div>
            
        </div>
    )
}