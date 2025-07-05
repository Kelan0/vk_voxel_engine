
#[macro_export]
macro_rules! log_error_and_anyhow {
    ($msg:literal $(, $arg:expr)*) => {{
        let formatted = format!($msg $(, $arg)*);
        error!("{}", formatted);
        anyhow!(formatted)
    }};
}

#[macro_export]
macro_rules! log_error_and_throw {
    ($err:expr, $msg:literal $(, $arg:expr)*) => {{
        error!($msg, $($arg)*);
        $err
    }};
}