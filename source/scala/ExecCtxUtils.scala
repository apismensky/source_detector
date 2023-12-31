/*
 * Copyright (C) from 2022 The Play Framework Contributors <https://github.com/playframework>, 2011-2021 Lightbend Inc. <https://www.lightbend.com>
 */

package play.utils

import scala.concurrent.ExecutionContext

// Workaround https://github.com/scala/bug/issues/7934#issuecomment-292425679
// deprecation warnings are not issued within deprecated methods/classes
@deprecated("", "") private[play] sealed class ExecCtxUtils {
  final def prepare(ec: ExecutionContext): ExecutionContext = ec.prepare()
}

/** INTERNAL API */
object ExecCtxUtils extends ExecCtxUtils
