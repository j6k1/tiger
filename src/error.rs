use std::{error, fmt, io};
use std::cell::{BorrowError, BorrowMutError};
use std::collections::VecDeque;
use std::num::{ParseFloatError, ParseIntError};
use std::sync::mpsc::{RecvError, RecvTimeoutError};
use std::sync::{MutexGuard, PoisonError};
use nncombinator::error::{ConfigReadError, CudaError, DeviceError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use packedsfen::error::ReadError;
use rayon::ThreadPoolBuildError;
use usiagent::error::{EventDispatchError, InfoSendError, LimitSizeError, PlayerError, SfenStringConvertError, UsiProtocolError};
use usiagent::event::{EventQueue, SystemEvent, SystemEventKind, UserEvent, UserEventKind};

#[derive(Debug)]
pub enum ApplicationError {
    StartupError(String),
    SfenStringConvertError(SfenStringConvertError),
    EventDispatchError(String),
    IOError(io::Error),
    ParseIntError(ParseIntError),
    ParseFloatError(ParseFloatError),
    ParseSfenError(ReadError),
    AgentRunningError(String),
    LogicError(String),
    InvalidStateError(String),
    LearningError(String),
    SerdeError(toml::ser::Error),
    ConfigReadError(ConfigReadError),
    InvalidSettingError(String),
    TrainingError(TrainingError),
    EvaluateError(EvaluateError),
    DeviceError(DeviceError),
    LayerInstantiationError(LayerInstantiationError),
    PersistenceError(PersistenceError),
    CudaError(CudaError),
    RecvError(RecvError),
    RecvTimeoutError(RecvTimeoutError),
    PoisonError(String),
    InfoSendError(InfoSendError),
    UsiProtocolError(UsiProtocolError),
    BorrowError(BorrowError),
    BorrowMutError(BorrowMutError),
    ThreadPoolBuildError(ThreadPoolBuildError),
    LimitSizeError(LimitSizeError)
}
impl fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ApplicationError::StartupError(ref s) => write!(f, "{}",s),
            ApplicationError::SfenStringConvertError(ref e) => write!(f, "{}",e),
            ApplicationError::EventDispatchError(ref s) => write!(f,"{}",s),
            ApplicationError::IOError(ref e) => write!(f, "{}",e),
            ApplicationError::ParseIntError(ref e) => write!(f, "{}",e),
            ApplicationError::ParseFloatError(ref e) => write!(f, "{}",e),
            ApplicationError::ParseSfenError(ref e) => write!(f,"{}",e),
            ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
            ApplicationError::LogicError(ref s) => write!(f,"{}",s),
            ApplicationError::InvalidStateError(ref s) => write!(f,"{}",s),
            ApplicationError::LearningError(ref s) => write!(f,"{}",s),
            ApplicationError::SerdeError(ref e) => write!(f,"{}",e),
            ApplicationError::InvalidSettingError(ref s) => write!(f,"{}",s),
            ApplicationError::ConfigReadError(ref e) => write!(f,"{}",e),
            ApplicationError::TrainingError(ref e) => write!(f,"{}",e),
            ApplicationError::EvaluateError(ref e) => write!(f,"{}",e),
            ApplicationError::DeviceError(ref e) => write!(f,"{}",e),
            ApplicationError::LayerInstantiationError(ref e) => write!(f,"{}",e),
            ApplicationError::PersistenceError(ref e) => write!(f,"{}",e),
            ApplicationError::CudaError(ref e) => write!(f, "An error occurred in the process of cuda. ({})",e),
            ApplicationError::RecvError(ref e) => write!(f, "{}",e),
            ApplicationError::RecvTimeoutError(ref e) => write!(f,"{}",e),
            ApplicationError::PoisonError(ref s) => write!(f,"{}",s),
            ApplicationError::InfoSendError(ref e) => write!(f,"{}",e),
            ApplicationError::UsiProtocolError(ref e) => write!(f,"{}",e),
            ApplicationError::BorrowError(ref e) => write!(f,"{}",e),
            ApplicationError::BorrowMutError(ref e) => write!(f,"{}",e),
            ApplicationError::ThreadPoolBuildError(ref e) => write!(f,"{}",e),
            ApplicationError::LimitSizeError(ref e) => write!(f,"{}",e)
        }
    }
}
impl error::Error for ApplicationError {
    fn description(&self) -> &str {
        match *self {
            ApplicationError::StartupError(_) => "Startup Error.",
            ApplicationError::SfenStringConvertError(_) => "An error occurred during conversion to sfen string.",
            ApplicationError::EventDispatchError(_) => "An error occurred while processing the event.",
            ApplicationError::IOError(_) => "IO Error.",
            ApplicationError::ParseIntError(_) => "An error occurred parsing the integer string.",
            ApplicationError::ParseFloatError(_) => "An error occurred parsing the float string.",
            ApplicationError::ParseSfenError(_) => "An error occurred parsing the packed sfen.",
            ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
            ApplicationError::LogicError(_) => "Logic error.",
            ApplicationError::InvalidStateError(_) => "Invalid state.",
            ApplicationError::LearningError(_) => "An error occurred while learning the neural network.",
            ApplicationError::SerdeError(_) => "An error occurred during serialization or deserialization.",
            ApplicationError::ConfigReadError(_) => "An error occurred while loading the neural network model.",
            ApplicationError::InvalidSettingError(_) => "Invalid setting.",
            ApplicationError::TrainingError(_) => "An error occurred while training the model.",
            ApplicationError::EvaluateError(_) => "An error occurred when running the neural network.",
            ApplicationError::DeviceError(_) => "An error occurred during device initialization.",
            ApplicationError::LayerInstantiationError(_) => "An unexpected error occurred during layer instantiation.",
            ApplicationError::PersistenceError(_) => "An error occurred when saving model information.",
            ApplicationError::CudaError(_) => "An error occurred in the process of cuda.",
            ApplicationError::RecvError(_) => "An error occurred while receiving the message.",
            ApplicationError::RecvTimeoutError(RecvTimeoutError::Disconnected) => "Disconnected while waiting for reception.",
            ApplicationError::RecvTimeoutError(RecvTimeoutError::Timeout) => "Timeout occurred while waiting to receive.",
            ApplicationError::PoisonError(_) => "panic occurred during thread execution.",
            ApplicationError::InfoSendError(_) => "An error occurred when sending info command.",
            ApplicationError::UsiProtocolError(_) => "An error occurred in the parsing or string generation process of string processing according to the USI protocol.",
            ApplicationError::BorrowError(_) => "already borrowed.",
            ApplicationError::BorrowMutError(_) => "already mutably borrowed.",
            ApplicationError::ThreadPoolBuildError(_) => "Failed to create thread pool.",
            ApplicationError::LimitSizeError(_) => "Size exceeds the upper limit."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ApplicationError::StartupError(_) => None,
            ApplicationError::SfenStringConvertError(ref e) => Some(e),
            ApplicationError::EventDispatchError(_) => None,
            ApplicationError::IOError(ref e) => Some(e),
            ApplicationError::ParseIntError(ref e) => Some(e),
            ApplicationError::ParseFloatError(ref e) => Some(e),
            ApplicationError::ParseSfenError(ref e) => Some(e),
            ApplicationError::AgentRunningError(_) => None,
            ApplicationError::LogicError(_) => None,
            ApplicationError::InvalidStateError(_) => None,
            ApplicationError::LearningError(_) => None,
            ApplicationError::SerdeError(ref e) => Some(e),
            ApplicationError::ConfigReadError(ref e) => Some(e),
            ApplicationError::InvalidSettingError(_) => None,
            ApplicationError::TrainingError(ref e) => Some(e),
            ApplicationError::EvaluateError(ref e) => Some(e),
            ApplicationError::DeviceError(ref e) => Some(e),
            ApplicationError::LayerInstantiationError(ref e) => Some(e),
            ApplicationError::PersistenceError(ref e) => Some(e),
            ApplicationError::CudaError(_) => None,
            ApplicationError::RecvError(ref e) => Some(e),
            ApplicationError::RecvTimeoutError(ref e) => Some(e),
            ApplicationError::PoisonError(_) => None,
            ApplicationError::InfoSendError(ref e) => Some(e),
            ApplicationError::UsiProtocolError(ref e) => Some(e),
            ApplicationError::BorrowError(ref e) => Some(e),
            ApplicationError::BorrowMutError(ref e) => Some(e),
            ApplicationError::ThreadPoolBuildError(ref e) => Some(e),
            ApplicationError::LimitSizeError(ref e) => Some(e),
        }
    }
}
impl PlayerError for ApplicationError {}
impl From<io::Error> for ApplicationError {
    fn from(err: io::Error) -> ApplicationError {
        ApplicationError::IOError(err)
    }
}
impl From<ParseIntError> for ApplicationError {
    fn from(err: ParseIntError) -> ApplicationError {
        ApplicationError::ParseIntError(err)
    }
}
impl From<ParseFloatError> for ApplicationError {
    fn from(err: ParseFloatError) -> ApplicationError {
        ApplicationError::ParseFloatError(err)
    }
}
impl From<ReadError> for ApplicationError {
    fn from(err: ReadError) -> ApplicationError {
        ApplicationError::ParseSfenError(err)
    }
}
impl From<SfenStringConvertError> for ApplicationError {
    fn from(err: SfenStringConvertError) -> ApplicationError {
        ApplicationError::SfenStringConvertError(err)
    }
}
impl<'a> From<EventDispatchError<'a,EventQueue<SystemEvent,SystemEventKind>,SystemEvent,ApplicationError>> for ApplicationError {
    fn from(err: EventDispatchError<'a, EventQueue<SystemEvent, SystemEventKind>, SystemEvent, ApplicationError>)
            -> ApplicationError {
        ApplicationError::EventDispatchError(format!("{}",err))
    }
}
impl<'a> From<EventDispatchError<'_, EventQueue<UserEvent, UserEventKind>, UserEvent, ApplicationError>> for ApplicationError {
    fn from(err: EventDispatchError<'_, EventQueue<UserEvent, UserEventKind>, UserEvent, ApplicationError>) -> Self {
        ApplicationError::EventDispatchError(format!("{}",err))
    }
}
impl From<toml::ser::Error> for ApplicationError {
    fn from(err: toml::ser::Error) -> ApplicationError {
        ApplicationError::SerdeError(err)
    }
}
impl From<ConfigReadError> for ApplicationError {
    fn from(err: ConfigReadError) -> ApplicationError {
        ApplicationError::ConfigReadError(err)
    }
}
impl From<TrainingError> for ApplicationError {
    fn from(err: TrainingError) -> ApplicationError {
        ApplicationError::TrainingError(err)
    }
}
impl From<EvaluateError> for ApplicationError {
    fn from(err: EvaluateError) -> ApplicationError {
        ApplicationError::EvaluateError(err)
    }
}
impl From<DeviceError> for ApplicationError {
    fn from(err: DeviceError) -> ApplicationError {
        ApplicationError::DeviceError(err)
    }
}
impl From<LayerInstantiationError> for ApplicationError {
    fn from(err: LayerInstantiationError) -> ApplicationError {
        ApplicationError::LayerInstantiationError(err)
    }
}
impl From<PersistenceError> for ApplicationError {
    fn from(err: PersistenceError) -> ApplicationError {
        ApplicationError::PersistenceError(err)
    }
}
impl From<CudaError> for ApplicationError {
    fn from(err: CudaError) -> ApplicationError {
        ApplicationError::CudaError(err)
    }
}
impl From<RecvError> for ApplicationError {
    fn from(err: RecvError) -> ApplicationError {
        ApplicationError::RecvError(err)
    }
}
impl From<RecvTimeoutError> for ApplicationError {
    fn from(err: RecvTimeoutError) -> Self {
        ApplicationError::RecvTimeoutError(err)
    }
}
impl From<PoisonError<MutexGuard<'_, VecDeque<std::sync::mpsc::Sender<()>>>>> for ApplicationError {
    fn from(err: PoisonError<MutexGuard<'_, VecDeque<std::sync::mpsc::Sender<()>>>>) -> ApplicationError {
        ApplicationError::PoisonError(format!("{}",err))
    }
}
impl From<PoisonError<MutexGuard<'_, ()>>> for ApplicationError {
    fn from(err: PoisonError<MutexGuard<'_, ()>>) -> ApplicationError {
        ApplicationError::PoisonError(format!("{}",err))
    }
}
impl From<ThreadPoolBuildError> for ApplicationError {
    fn from(err: ThreadPoolBuildError) -> ApplicationError {
        ApplicationError::ThreadPoolBuildError(err)
    }
}
impl From<InfoSendError> for ApplicationError {
    fn from(err: InfoSendError) -> ApplicationError {
        ApplicationError::InfoSendError(err)
    }
}
impl From<LimitSizeError> for ApplicationError {
    fn from(err: LimitSizeError) -> ApplicationError {
        ApplicationError::LimitSizeError(err)
    }
}
impl From<UsiProtocolError> for ApplicationError {
    fn from(err: UsiProtocolError) -> ApplicationError {
        ApplicationError::UsiProtocolError(err)
    }
}
impl From<BorrowError> for ApplicationError {
    fn from(err: BorrowError) -> ApplicationError {
        ApplicationError::BorrowError(err)
    }
}
impl From<BorrowMutError> for ApplicationError {
    fn from(err: BorrowMutError) -> ApplicationError {
        ApplicationError::BorrowMutError(err)
    }
}
#[derive(Debug)]
pub enum EvaluationError {
    InternalError(ApplicationError),
    Timeout,
}
impl From<ApplicationError> for EvaluationError {
    fn from(err: ApplicationError) -> EvaluationError {
        EvaluationError::InternalError(err)
    }
}
